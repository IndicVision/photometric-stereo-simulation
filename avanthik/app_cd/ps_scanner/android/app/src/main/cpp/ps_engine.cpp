#include <stdint.h>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "libraw/libraw.h"
#include <android/log.h>

#define LOG_TAG "PS_ENGINE"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

extern "C" __attribute__((visibility("default"))) __attribute__((used))
int32_t
processAndDenoise(const char **filePaths, int numFiles, const char *outputPath)
{
  if (numFiles <= 0)
  {
    LOGE("No files provided to the engine.");
    return -200;
  }

  cv::Mat accumulator;
  LibRaw RawProcessor;

  // Configure LibRaw for 16-bit linear data extraction
  RawProcessor.imgdata.params.output_bps = 16;
  RawProcessor.imgdata.params.gamm[0] = 1.0;
  RawProcessor.imgdata.params.gamm[1] = 1.0;
  RawProcessor.imgdata.params.no_auto_bright = 1;
  RawProcessor.imgdata.params.use_camera_wb = 1;
  RawProcessor.imgdata.params.user_qual = 0; // Fast linear

  for (int i = 0; i < numFiles; i++)
  {
    LOGI("Engine opening: %s", filePaths[i]);

    int ret = RawProcessor.open_file(filePaths[i]);
    if (ret != LIBRAW_SUCCESS)
    {
      LOGE("LibRaw open_file failed for %s with error %d", filePaths[i], ret);
      return -201;
    }

    if (RawProcessor.unpack() != LIBRAW_SUCCESS)
      return -202;
    if (RawProcessor.dcraw_process() != LIBRAW_SUCCESS)
      return -203;

    libraw_processed_image_t *image = RawProcessor.dcraw_make_mem_image(&ret);
    if (!image)
      return -204;

    // Create 16-bit 3-channel Mat from LibRaw buffer
    cv::Mat currentFrame(image->height, image->width, CV_16UC3, image->data);

    // Convert to 32-bit Float so we can sum them up without losing precision
    cv::Mat floatFrame;
    currentFrame.convertTo(floatFrame, CV_32FC3);

    if (accumulator.empty())
    {
      accumulator = floatFrame.clone();
    }
    else
    {
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

  // 3. Save to storage
  // imwrite detects the .png extension and saves 16-bit correctly
  LOGI("Saving denoised result to: %s", outputPath);
  bool success = cv::imwrite(outputPath, finalImage);

  return success ? 1 : -205;
}