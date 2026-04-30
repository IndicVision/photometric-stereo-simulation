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

// ─────────────────────────────────────────────────────────────────────────────
// Helper: apply the same radiometrically-linear LibRaw flags used across both
// functions. Centralised here so the two functions cannot drift out of sync.
// ─────────────────────────────────────────────────────────────────────────────
static void applyLinearFlags(LibRaw& rp) {
    rp.imgdata.params.output_bps        = 16;  // 16-bit output
    rp.imgdata.params.gamm[0]           = 1.0; // no gamma: linear light
    rp.imgdata.params.gamm[1]           = 1.0;
    rp.imgdata.params.no_auto_bright    = 1;   // no automatic brightness
    rp.imgdata.params.use_camera_wb     = 0;   // no per-frame WB multipliers
    rp.imgdata.params.user_qual         = 0;   // fast bilinear demosaic
    rp.imgdata.params.no_auto_scale     = 1;   // disable per-frame histogram stretch
    rp.imgdata.params.use_camera_matrix = 0;   // no perceptual colour mixing
    rp.imgdata.params.highlight         = 0;   // clean clip, no reconstruction
}

// ─────────────────────────────────────────────────────────────────────────────
// computeProbeMax
//
// Purpose: given N RAW DNG file paths (one per light, probe captures), return
// the single highest Nth-percentile raw ADU value seen across all frames.
// This value is used by the Kotlin auto-exposure logic to compute the shutter
// speed that places the scene near 85% of sensor full-scale.
//
// Key design decisions:
//   • Operates on pre-demosaic Bayer data (raw_image, uint16_t).  Demosaicing
//     interpolates neighbours, which can inflate bright values at hot pixels.
//     Working on the native Bayer avoids that smearing.
//   • Uses a 65536-bucket histogram + CDF walk instead of a pixel-level sort.
//     For a 12 MP Bayer array this is O(N) time and costs only 512 KB per call.
//   • The black-level override is NOT applied here; we are looking for the
//     brightest pixel relative to the sensor floor, so including it is fine.
//     The floor itself is well below any scene brightness we care about.
//   • Returns -1.0 if every file failed, so Kotlin can detect and fall back to
//     manual exposure gracefully.
// ─────────────────────────────────────────────────────────────────────────────
extern "C" JNIEXPORT jfloat JNICALL
Java_com_ps_photocapture_MainActivity_computeProbeMax(
        JNIEnv* env,
        jobject /* this */,
        jobjectArray filePaths,
        jint numFiles,
        jfloat percentile)   // e.g. 99.0 for 99th percentile
{
    if (numFiles <= 0) {
        LOGE("computeProbeMax: no files provided.");
        return -1.0f;
    }

    std::vector<std::string> paths;
    for (int i = 0; i < numFiles; i++) {
        jstring jPath = (jstring) env->GetObjectArrayElement(filePaths, i);
        const char* pathChars = env->GetStringUTFChars(jPath, nullptr);
        paths.push_back(std::string(pathChars));
        env->ReleaseStringUTFChars(jPath, pathChars);
        env->DeleteLocalRef(jPath);
    }

    LibRaw RawProcessor;
    applyLinearFlags(RawProcessor);

    float globalMax = -1.0f;

    for (int i = 0; i < numFiles; i++) {
        LOGI("computeProbeMax: opening %s", paths[i].c_str());

        if (RawProcessor.open_file(paths[i].c_str()) != LIBRAW_SUCCESS) {
            LOGE("computeProbeMax: open_file failed for %s. Skipping.", paths[i].c_str());
            continue;
        }
        if (RawProcessor.unpack() != LIBRAW_SUCCESS) {
            LOGE("computeProbeMax: unpack failed for %s. Skipping.", paths[i].c_str());
            RawProcessor.recycle();
            continue;
        }

        // Access native Bayer array directly — no demosaic step needed.
        uint16_t* rawImage = RawProcessor.imgdata.rawdata.raw_image;
        int rawW = RawProcessor.imgdata.sizes.raw_width;
        int rawH = RawProcessor.imgdata.sizes.raw_height;

        if (!rawImage || rawW == 0 || rawH == 0) {
            LOGE("computeProbeMax: raw_image is null or zero-size for %s. Skipping.",
                 paths[i].c_str());
            RawProcessor.recycle();
            continue;
        }

        long totalPixels = (long)rawW * rawH;

        // Build histogram over all 65536 possible uint16 values. At 12 MP
        // Bayer (48 M uint16 values in a 16 MP sensor) this runs in < 50 ms.
        std::vector<long> hist(65536, 0L);
        for (long p = 0; p < totalPixels; p++) {
            hist[(int)rawImage[p]]++;
        }

        // Walk the CDF to find the value at `percentile`%.
        long threshold = (long)(percentile * 0.01f * (float)totalPixels);
        long cumulative = 0L;
        float framePercentileVal = 0.0f;
        for (int v = 0; v < 65536; v++) {
            cumulative += hist[v];
            if (cumulative >= threshold) {
                framePercentileVal = (float)v;
                break;
            }
        }

        LOGI("computeProbeMax: %s -> %.0f at %.1f%%",
             paths[i].c_str(), framePercentileVal, percentile);

        if (framePercentileVal > globalMax) {
            globalMax = framePercentileVal;
        }

        RawProcessor.recycle();
    }

    LOGI("computeProbeMax: global max at %.1f%% percentile = %.0f", percentile, globalMax);
    return globalMax;
}

// ─────────────────────────────────────────────────────────────────────────────
// processAndDenoise  (unchanged from previous version)
// ─────────────────────────────────────────────────────────────────────────────
extern "C" JNIEXPORT jint JNICALL
Java_com_ps_photocapture_MainActivity_processAndDenoise(
        JNIEnv* env,
        jobject /* this */,
        jobjectArray filePaths,
        jint numFiles,
        jstring outputPath,
        jint rotationDegrees,
        jfloat scalingFactor)
{
    if (numFiles <= 0) {
        LOGE("No files provided to the engine.");
        return -200;
    }

    std::vector<std::string> paths;
    for (int i = 0; i < numFiles; i++) {
        jstring jPath = (jstring) env->GetObjectArrayElement(filePaths, i);
        const char* pathChars = env->GetStringUTFChars(jPath, nullptr);
        paths.push_back(std::string(pathChars));
        env->ReleaseStringUTFChars(jPath, pathChars);
        env->DeleteLocalRef(jPath);
    }

    const char* outPathChars = env->GetStringUTFChars(outputPath, nullptr);
    std::string outPath(outPathChars);
    env->ReleaseStringUTFChars(outputPath, outPathChars);

    cv::Mat accumulator;
    LibRaw RawProcessor;

    applyLinearFlags(RawProcessor);

    int validFrames = 0;

    for (int i = 0; i < numFiles; i++) {
        LOGI("Engine opening: %s", paths[i].c_str());

        int ret = RawProcessor.open_file(paths[i].c_str());
        if (ret != LIBRAW_SUCCESS) {
            LOGE("LibRaw open_file failed for %s with error %d. Skipping frame.",
                 paths[i].c_str(), ret);
            continue;
        }
        if (RawProcessor.unpack() != LIBRAW_SUCCESS) {
            LOGE("LibRaw unpack failed for %s. Skipping frame.", paths[i].c_str());
            RawProcessor.recycle();
            continue;
        }

        // Black level lock: prevent thermal drift between frames.
        for (int c = 0; c < 4; c++) {
            RawProcessor.imgdata.color.black_stat[c] = 0;
        }
        RawProcessor.imgdata.color.black = 0;

        if (RawProcessor.dcraw_process() != LIBRAW_SUCCESS) {
            LOGE("LibRaw dcraw_process failed for %s. Skipping frame.", paths[i].c_str());
            RawProcessor.recycle();
            continue;
        }

        libraw_processed_image_t *image = RawProcessor.dcraw_make_mem_image(&ret);
        if (!image) {
            LOGE("LibRaw mem image allocation failed for %s. Skipping frame.",
                 paths[i].c_str());
            RawProcessor.recycle();
            continue;
        }

        cv::Mat currentFrame(image->height, image->width, CV_16UC3, image->data);
        cv::Mat floatFrame;
        currentFrame.convertTo(floatFrame, CV_32FC3);

        if (accumulator.empty()) {
            accumulator = floatFrame.clone();
        } else {
            accumulator += floatFrame;
        }

        validFrames++;
        LibRaw::dcraw_clear_mem(image);
        RawProcessor.recycle();
    }

    if (validFrames == 0) {
        LOGE("All provided frames failed to process.");
        return -204;
    }

    accumulator /= (float)validFrames;
    accumulator *= scalingFactor;

    cv::Mat finalImage;
    accumulator.convertTo(finalImage, CV_16UC3);

    // LibRaw gives RGB; OpenCV imwrite expects BGR.
    cv::cvtColor(finalImage, finalImage, cv::COLOR_RGB2BGR);

    if (rotationDegrees == 90) {
        cv::rotate(finalImage, finalImage, cv::ROTATE_90_CLOCKWISE);
    } else if (rotationDegrees == 180) {
        cv::rotate(finalImage, finalImage, cv::ROTATE_180);
    } else if (rotationDegrees == 270) {
        cv::rotate(finalImage, finalImage, cv::ROTATE_90_COUNTERCLOCKWISE);
    }

    LOGI("Saving result to: %s (averaged from %d valid frames)", outPath.c_str(), validFrames);
    bool success = cv::imwrite(outPath, finalImage);
    // 2. Generate the 8-bit preview path (replace .png with _preview.png)
    std::string previewPath = outPath;
    size_t dotPos = previewPath.find_last_of('.');
    if (dotPos != std::string::npos) {
        previewPath.insert(dotPos, "_preview");
    } else {
        previewPath += "_preview.png"; // Fallback just in case
    }
    // 3. Convert CV_16UC3 to CV_8UC3 (scale by 1/256) and save
    cv::Mat previewImage;
    finalImage.convertTo(previewImage, CV_8UC3, 1.0 / 256.0);
    cv::imwrite(previewPath, previewImage);

    return success ? 1 : -205;
}
