#include <jni.h>
#include <stdint.h>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "libraw/libraw.h"
#include <android/log.h>

#define LOG_TAG "PS_ENGINE"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// --- THIS IS THE NEW KOTLIN JNI DOORWAY ---
extern "C" JNIEXPORT jint JNICALL
Java_com_ps_photocapture_MainActivity_processAndDenoise(
        JNIEnv* env,
        jobject /* this */,
        jobjectArray filePaths,
        jint numFiles,
        jstring outputPath,
        jint rotationDegrees) // <--- NEW PARAMETER
{
  if (numFiles <= 0) {
    LOGE("No files provided to the engine.");
    return -200;
  }

  // 1. Convert Kotlin String Array to C++ std::vector
  std::vector<std::string> paths;
  for (int i = 0; i < numFiles; i++) {
    jstring jPath = (jstring) env->GetObjectArrayElement(filePaths, i);
    const char* pathChars = env->GetStringUTFChars(jPath, nullptr);
    paths.push_back(std::string(pathChars));
    env->ReleaseStringUTFChars(jPath, pathChars);
    env->DeleteLocalRef(jPath);
  }

  // 2. Convert Kotlin Output Path String to C++ String
  const char* outPathChars = env->GetStringUTFChars(outputPath, nullptr);
  std::string outPath(outPathChars);
  env->ReleaseStringUTFChars(outputPath, outPathChars);

  cv::Mat accumulator;
  LibRaw RawProcessor;

  // Configure LibRaw for 16-bit linear data extraction
  RawProcessor.imgdata.params.output_bps = 16;
  RawProcessor.imgdata.params.gamm[0] = 1.0;
  RawProcessor.imgdata.params.gamm[1] = 1.0;
  RawProcessor.imgdata.params.no_auto_bright = 1;
  RawProcessor.imgdata.params.use_camera_wb = 1;
  RawProcessor.imgdata.params.user_qual = 0; // Fast linear

  for (int i = 0; i < numFiles; i++) {
    LOGI("Engine opening: %s", paths[i].c_str());

    int ret = RawProcessor.open_file(paths[i].c_str());
    if (ret != LIBRAW_SUCCESS) {
      LOGE("LibRaw open_file failed for %s with error %d", paths[i].c_str(), ret);
      return -201;
    }

    if (RawProcessor.unpack() != LIBRAW_SUCCESS) return -202;
    if (RawProcessor.dcraw_process() != LIBRAW_SUCCESS) return -203;

    libraw_processed_image_t *image = RawProcessor.dcraw_make_mem_image(&ret);
    if (!image) return -204;

    // Create 16-bit 3-channel Mat from LibRaw buffer
    cv::Mat currentFrame(image->height, image->width, CV_16UC3, image->data);

    // Convert to 32-bit Float so we can sum them up without losing precision
    cv::Mat floatFrame;
    currentFrame.convertTo(floatFrame, CV_32FC3);

    if (accumulator.empty()) {
      accumulator = floatFrame.clone();
    } else {
      // Accumulate (Denoising step)
      accumulator += floatFrame;
    }

    // Cleanup memory for this frame
    LibRaw::dcraw_clear_mem(image);
    RawProcessor.recycle();
  }

  // 1. Calculate the average (Divide sum by 4)
  accumulator /= (float)numFiles;

  // 2. Convert from Float back to 16-bit Integer for the PNG format
  cv::Mat finalImage;
  accumulator.convertTo(finalImage, CV_16UC3);

  // --- THE COLOR FIX ---
  // LibRaw gives us RGB, but OpenCV's imwrite expects BGR.
  // We must swap the channels before saving so the PNG colors are correct!
  cv::cvtColor(finalImage, finalImage, cv::COLOR_RGB2BGR);


  // --- THE DYNAMIC ORIENTATION FIX ---
  // We use the exact angle passed from the Android hardware
  if (rotationDegrees == 90) {
    cv::rotate(finalImage, finalImage, cv::ROTATE_90_CLOCKWISE);
  } else if (rotationDegrees == 180) {
    cv::rotate(finalImage, finalImage, cv::ROTATE_180);
  } else if (rotationDegrees == 270) {
    cv::rotate(finalImage, finalImage, cv::ROTATE_90_COUNTERCLOCKWISE);
  }
  // If it's 0, we do nothing!


  // 3. Save to storage
  LOGI("Saving denoised result to: %s", outPath.c_str());
  bool success = cv::imwrite(outPath, finalImage);

  return success ? 1 : -205;
}