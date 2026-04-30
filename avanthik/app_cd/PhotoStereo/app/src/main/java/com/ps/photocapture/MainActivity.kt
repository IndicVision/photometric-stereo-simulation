package com.ps.photocapture

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.RectF
import android.graphics.SurfaceTexture
import android.graphics.Typeface
import android.hardware.camera2.*
import android.hardware.camera2.params.OutputConfiguration
import android.hardware.camera2.params.SessionConfiguration
import android.media.Image
import android.media.ImageReader
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.os.Looper
import android.text.InputType
import android.view.Surface
import android.view.TextureView
import android.view.View
import android.widget.*
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.widget.SwitchCompat
import androidx.core.app.ActivityCompat
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.asRequestBody
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.io.FileWriter
import java.util.Locale
import java.util.concurrent.Executors
import androidx.core.graphics.get
import androidx.core.graphics.scale
import java.util.zip.ZipInputStream
import java.io.FileInputStream


import androidx.activity.result.contract.ActivityResultContracts
import androidx.documentfile.provider.DocumentFile


@SuppressLint("SetTextI18n")
class MainActivity : AppCompatActivity() {

    // ── Configuration Arrays ───────────────────────────────────────────
    private val isoValues = intArrayOf(50, 100, 200, 400, 800, 1600, 3200)
    private val shutterNsValues = longArrayOf(1_000_000L, 2_000_000L, 4_000_000L, 8_000_000L, 10_000_000L, 16_666_666L, 20_000_000L, 33_333_333L)
    private val shutterLabels = arrayOf("1/1000s","1/500s","1/250s","1/125s","1/100s","1/60s","1/50s","1/30s")
    private val wbModeValues = intArrayOf(
        CameraMetadata.CONTROL_AWB_MODE_AUTO, CameraMetadata.CONTROL_AWB_MODE_INCANDESCENT,
        CameraMetadata.CONTROL_AWB_MODE_FLUORESCENT, CameraMetadata.CONTROL_AWB_MODE_DAYLIGHT,
        CameraMetadata.CONTROL_AWB_MODE_CLOUDY_DAYLIGHT, CameraMetadata.CONTROL_AWB_MODE_SHADE
    )
    private val wbLabels = arrayOf("Auto","Incandescent","Fluorescent","Daylight","Cloudy","Shade")

    // ── Internal State ────────────────────────────────────────────────
    private var currentIsoIdx     = 1
    private var currentShutterIdx = 2
    private var currentWbIdx      = 3   // Daylight default
    private var formatMode        = "BOTH"
    private var framesPerLight    = 1

    private var isManualFocusMode   = false
    private var manualFocusDistance = 0.0f
    private var minHardwareFocusDistance = 0.0f
    private var maxObservedDiopter       = 0.0f

    private var focusLocked         = false
    private var lockedFocusDistance = 0.0f
    private var isFocusing          = false
    private var timeoutRunnable: Runnable? = null
    private var histFrameCount      = 0

    private var isCapturePhase = false
    private var isJetsonConnected = false


    @Volatile private var cancelNetworkTask = false
    private var currentOkHttpCall: Call? = null

    private var isPreviewLightOn  = false

    // ── Capture Loop State ────────────────────────────────────────────
    private var currentLight   = 0
    private var captureCount   = 0
    private var isCapturing    = false
    private var lightStartTime = 0L

    private var rawWidth = 0
    private var rawHeight = 0
    private var pendingJpegBytes: ByteArray? = null
    private var pendingRawImage:  Image?     = null
    private var pendingCaptureResult: TotalCaptureResult? = null

    // ── UI Components ─────────────────────────────────────────────────
    private lateinit var connectionStatus:   TextView
    private lateinit var textureView:      TextureView
    private lateinit var focusOverlay:     FocusOverlayView
    private lateinit var cameraInfoText:   TextView

    private lateinit var setupUiContainer: ScrollView
    private lateinit var isoSeekBar:       SeekBar
    private lateinit var isoValueText:     TextView
    private lateinit var shutterSeekBar:   SeekBar
    private lateinit var shutterValueText: TextView
    private lateinit var wbSpinner:        Spinner
    private lateinit var focusModeSwitch:  SwitchCompat
    private lateinit var manualFocusContainer: LinearLayout
    private lateinit var manualFocusSeekBar:   SeekBar
    private lateinit var manualFocusValueText: TextView
    private lateinit var focusStatusText:  TextView
    private lateinit var lockFocusButton:   Button
    private lateinit var formatRadioGroup:  RadioGroup
    private lateinit var framesInput:       EditText
    private lateinit var continueToCaptureButton: Button

    private lateinit var captureUiContainer:  LinearLayout
    private lateinit var statusText:          TextView
    private lateinit var startSequenceButton: Button
    private lateinit var uploadButton:        Button
    private lateinit var retakeButton:        Button
    private lateinit var previewScrollView:   HorizontalScrollView
    private lateinit var previewContainer:    LinearLayout



    private lateinit var cbLight1: CheckBox
    private lateinit var cbLight2: CheckBox
    private lateinit var cbLight3: CheckBox
    private lateinit var cbLight4: CheckBox
    private lateinit var btnTurnOffPreview: Button

    // ── Camera Core ────────────────────────────────────────────────────
    private var cameraDevice:         CameraDevice? = null
    private var cameraCaptureSession: CameraCaptureSession? = null
    private var backgroundThread:     HandlerThread? = null
    private var backgroundHandler:    Handler? = null
    private var previewSurface:       Surface? = null
    private lateinit var jpegReader:  ImageReader
    private lateinit var rawReader:   ImageReader
    private var previewBuilder: CaptureRequest.Builder? = null

    // ── WebSocket & Storage ───────────────────────────────────────────
    private var webSocket: WebSocket? = null
    // Dedicated handler for WS retries — always alive, never null
    private val wsRetryHandler = Handler(Looper.getMainLooper())
    private val finalPreviewImages = mutableListOf<String>()
    private val saveFolder by lazy { getExternalFilesDir(null)!!.absolutePath + "/PS" }



    // ── Folder Picker Launcher ─────────────────────────────────────────
    private val directoryPickerLauncher = registerForActivityResult(ActivityResultContracts.OpenDocumentTree()) { uri ->
        if (uri != null) {
            extractZipToUserFolder(uri)
        } else {
            updateStatus("❌ Save cancelled. Files are in app cache.")
            runOnUiThread { uploadButton.isEnabled = true }
        }
    }



    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)


        // --- NEW LINE: Keep the screen awake during long processing ---
        window.addFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        bindViews()
        connectToJetson()
        setupISO()
        setupShutter()
        setupWB()
        setupManualFocusUI()
        setupFocusButton()
        setupPreviewLightingUI() // <-- ADD THIS LINE HERE

        continueToCaptureButton.setOnClickListener { enterCapturePhase() }
        startSequenceButton.setOnClickListener { startCaptureSequence() }
        retakeButton.setOnClickListener { performFullRetakeReset() }

        uploadButton.setOnClickListener {
            uploadButton.isEnabled = false
            runBatchProcess()
        }

        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), CAMERA_REQUEST_CODE)
        } else {
            startCamera()
        }
    }

    private fun bindViews() {
        textureView      = findViewById(R.id.textureView)
        connectionStatus = findViewById(R.id.connectionStatus)
        focusOverlay     = findViewById(R.id.focusOverlay)
        cameraInfoText   = findViewById(R.id.cameraInfoText)

        setupUiContainer = findViewById(R.id.setupUiContainer)
        isoSeekBar       = findViewById(R.id.isoSeekBar)
        isoValueText     = findViewById(R.id.isoValueText)
        shutterSeekBar   = findViewById(R.id.shutterSeekBar)
        shutterValueText = findViewById(R.id.shutterValueText)
        wbSpinner        = findViewById(R.id.wbSpinner)
        focusModeSwitch  = findViewById(R.id.focusModeSwitch)
        manualFocusContainer = findViewById(R.id.manualFocusContainer)
        manualFocusSeekBar   = findViewById(R.id.manualFocusSeekBar)
        manualFocusValueText = findViewById(R.id.manualFocusValueText)
        focusStatusText  = findViewById(R.id.focusStatusText)
        lockFocusButton  = findViewById(R.id.lockFocusButton)
        formatRadioGroup = findViewById(R.id.formatRadioGroup)
        framesInput      = findViewById(R.id.framesInput)
        continueToCaptureButton = findViewById(R.id.continueToCaptureButton)



        // Add these 5 lines:
        cbLight1 = findViewById(R.id.cbLight1)
        cbLight2 = findViewById(R.id.cbLight2)
        cbLight3 = findViewById(R.id.cbLight3)
        cbLight4 = findViewById(R.id.cbLight4)
        btnTurnOffPreview = findViewById(R.id.btnTurnOffPreview)


        captureUiContainer  = findViewById(R.id.captureUiContainer)
        statusText          = findViewById(R.id.statusText)
        startSequenceButton = findViewById(R.id.startSequenceButton)
        uploadButton        = findViewById(R.id.uploadButton)
        retakeButton        = findViewById(R.id.retakeButton)
        previewScrollView   = findViewById(R.id.previewScrollView)
        previewContainer    = findViewById(R.id.previewContainer)

        continueToCaptureButton.isEnabled = false
    }

    // ── Continue Button Gate ──────────────────────────────────────────
    private fun updateContinueButton() {
        val focusReady = isManualFocusMode || focusLocked
        runOnUiThread { continueToCaptureButton.isEnabled = isJetsonConnected && focusReady }
    }


    private fun performFullRetakeReset() {
        updateStatus("🧹 Factory Resetting...")


        cancelNetworkTask = true // <--- KILLS ANY BACKGROUND UPLOADS/DOWNLOADS
        currentOkHttpCall?.cancel() // <--- ADD THIS: Instantly severs the 200MB upload transfer!

        // 1. Wipe all tracking lists and capture variables
        finalPreviewImages.clear()
        captureCount = 0
        currentLight = 0
        isCapturing = false

        // 2. Clear any hanging raw images in memory
        pendingJpegBytes = null
        pendingRawImage?.close()
        pendingRawImage = null
        pendingCaptureResult = null

        // 3. HARD RESET THE UI & BUTTON LOGIC
        runOnUiThread {
            previewContainer.removeAllViews()
            previewScrollView.visibility = View.GONE
            textureView.visibility = View.VISIBLE

            // Reset the Upload button back to its original job (Convert PNGs)
            uploadButton.visibility = View.GONE
            uploadButton.isEnabled = true
            uploadButton.text = "PROCESS RAW IMAGES"
            uploadButton.setBackgroundColor(android.graphics.Color.parseColor("#FF9800")) // Orange
            uploadButton.setOnClickListener {
                uploadButton.isEnabled = false
                runBatchProcess()
            }

            // Reset the Sequence button
            startSequenceButton.visibility = View.VISIBLE
            startSequenceButton.isEnabled = true
            statusText.text = "Ready for Capture"

            // Go back to the visual Setup Phase
            enterSetupPhase()
        }

        // 4. Force all physical lights off
        sendPreviewLightOff()

        // 5. Delete old files from the Android directory so they don't mix
        Thread {
            val dir = File(saveFolder)
            if (dir.exists()) {
                dir.listFiles()?.forEach { it.deleteRecursively() }
            }
        }.start()
    }



    // ── Phase Transitions ─────────────────────────────────────────────

    private fun enterCapturePhase() {
        isCapturePhase = true
        setupUiContainer.visibility = View.GONE
        captureUiContainer.visibility = View.VISIBLE
        startSequenceButton.visibility = View.VISIBLE
        startSequenceButton.isEnabled = true
        uploadButton.visibility = View.GONE

        updatePreviewRequest()
        updateInfoText()
    }

    private fun enterSetupPhase() {
        // Keep WebSocket to Jetson alive — do not close it here

        isCapturePhase = false
        captureUiContainer.visibility = View.GONE
        previewScrollView.visibility = View.GONE
        textureView.visibility = View.VISIBLE
        setupUiContainer.visibility = View.VISIBLE
        focusOverlay.visibility = if (isManualFocusMode) View.GONE else View.VISIBLE



        updatePreviewRequest()
        updateInfoText()
    }

    // ── UI Listeners ──────────────────────────────────────────────────
    private fun setupPreviewLightingUI() {
        val checkBoxes = listOf(cbLight1, cbLight2, cbLight3, cbLight4)

        val checkListener = android.widget.CompoundButton.OnCheckedChangeListener { _, _ ->
            val selectedLights = mutableListOf<Int>()
            checkBoxes.forEachIndexed { index, checkBox ->
                if (checkBox.isChecked) selectedLights.add(index + 1)
            }

            if (selectedLights.isEmpty()) {
                sendPreviewLightOff()
            } else {
                sendToJetson("SET_PREVIEW:${selectedLights.joinToString("")}")
                isPreviewLightOn = true
                updateInfoText()
            }
        }

        checkBoxes.forEach { it.setOnCheckedChangeListener(checkListener) }

        btnTurnOffPreview.setOnClickListener {
            // Uncheck all boxes visually (this triggers the listener to turn them off)
            checkBoxes.forEach { it.isChecked = false }
        }
    }






    private fun setupISO() {
        isoSeekBar.max = isoValues.size - 1
        isoSeekBar.progress = currentIsoIdx
        isoValueText.text = "ISO ${isoValues[currentIsoIdx]}"
        isoSeekBar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(sb: SeekBar, p: Int, fromUser: Boolean) {
                currentIsoIdx = p
                isoValueText.text = "ISO ${isoValues[p]}"
                if (!isFocusing) updatePreviewRequest()
                updateInfoText()
            }
            override fun onStartTrackingTouch(sb: SeekBar) {}
            override fun onStopTrackingTouch(sb: SeekBar) {}
        })
    }

    private fun setupShutter() {
        shutterSeekBar.max = shutterNsValues.size - 1
        shutterSeekBar.progress = currentShutterIdx
        shutterValueText.text = shutterLabels[currentShutterIdx]
        shutterSeekBar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(sb: SeekBar, p: Int, fromUser: Boolean) {
                currentShutterIdx = p
                shutterValueText.text = shutterLabels[p]
                if (!isFocusing) updatePreviewRequest()
                updateInfoText()
            }
            override fun onStartTrackingTouch(sb: SeekBar) {}
            override fun onStopTrackingTouch(sb: SeekBar) {}
        })
    }

    private fun setupWB() {
        // Create a custom adapter to force the text colors to be readable
        val adapter = object : ArrayAdapter<String>(this, android.R.layout.simple_spinner_item, wbLabels) {

            // 1. This controls the text color of the CURRENTLY SELECTED item (the closed spinner)
            override fun getView(position: Int, convertView: View?, parent: android.view.ViewGroup): View {
                val view = super.getView(position, convertView, parent) as TextView
                view.setTextColor(android.graphics.Color.parseColor("#00FF00")) // Bright Green
                view.textSize = 14f
                view.gravity = android.view.Gravity.END
                return view
            }

            // 2. This controls the text color of the DROPDOWN MENU when you tap it
            override fun getDropDownView(position: Int, convertView: View?, parent: android.view.ViewGroup): View {
                val view = super.getDropDownView(position, convertView, parent) as TextView
                view.setTextColor(android.graphics.Color.BLACK) // Black text for the white dropdown box
                view.setPadding(30, 30, 30, 30) // Make it easier to tap
                return view
            }
        }

        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        wbSpinner.adapter = adapter
        wbSpinner.setSelection(3) // Daylight default

        wbSpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(p: AdapterView<*>, v: View?, pos: Int, id: Long) {
                currentWbIdx = pos
                if (!isFocusing) updatePreviewRequest()
                updateInfoText()
            }
            override fun onNothingSelected(p: AdapterView<*>) {}
        }
    }

    private fun setupManualFocusUI() {
        focusModeSwitch.setOnCheckedChangeListener { _, isChecked ->
            isManualFocusMode = isChecked
            if (isChecked) {
                manualFocusContainer.visibility = View.VISIBLE
                findViewById<View>(R.id.autoFocusControls).visibility = View.GONE
                focusOverlay.visibility = View.GONE
                updateContinueButton()
            } else {
                manualFocusContainer.visibility = View.GONE
                findViewById<View>(R.id.autoFocusControls).visibility = View.VISIBLE
                focusOverlay.visibility = View.VISIBLE
                focusLocked = false
                lockFocusButton.text = getString(R.string.btn_lock_focus)
                focusStatusText.text = getString(R.string.focus_status_auto)
                updateContinueButton()
                focusOverlay.resetFocus()
            }
            if (!isFocusing) updatePreviewRequest()
        }

        manualFocusSeekBar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                if (fromUser) {
                    val overdriveLimit = minHardwareFocusDistance * 1.5f
                    val percentage = progress / 1000f
                    manualFocusDistance = percentage * overdriveLimit
                    updatePreviewRequest()
                }
            }
            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })
    }

    private fun setupFocusButton() {
        lockFocusButton.setOnClickListener {
            if (focusLocked) {
                focusLocked = false
                lockedFocusDistance = 0.0f
                focusStatusText.text = getString(R.string.focus_status_auto)
                lockFocusButton.text = getString(R.string.btn_lock_focus)
                updateContinueButton()
                focusOverlay.resetFocus()
                updatePreviewRequest()
            } else {
                triggerAFLock()
            }
        }
    }

    // ── Camera Callbacks ──────────────────────────────────────────────
    private val previewCallback = object : CameraCaptureSession.CaptureCallback() {
        override fun onCaptureCompleted(session: CameraCaptureSession, request: CaptureRequest, result: TotalCaptureResult) {
            histFrameCount++
            if (histFrameCount % HIST_SAMPLE_EVERY == 0) {
                sampleHistogramFromPreview()
            }

            val currentDistance = result.get(CaptureResult.LENS_FOCUS_DISTANCE) ?: 0.0f
            if (currentDistance > maxObservedDiopter) maxObservedDiopter = currentDistance

            if (isManualFocusMode && !isCapturePhase) {
                runOnUiThread {
                    val d = if (currentDistance <= 0.0f) "∞" else String.format(Locale.US, "%.2f Diopters", currentDistance)
                    manualFocusValueText.text = "Focus Distance: $d"
                }
            }

            if (isFocusing) {
                val afState = result.get(CaptureResult.CONTROL_AF_STATE) ?: return
                if (afState == CaptureResult.CONTROL_AF_STATE_FOCUSED_LOCKED ||
                    afState == CaptureResult.CONTROL_AF_STATE_NOT_FOCUSED_LOCKED) {

                    isFocusing = false
                    timeoutRunnable?.let { backgroundHandler?.removeCallbacks(it) }

                    if (afState == CaptureResult.CONTROL_AF_STATE_FOCUSED_LOCKED) {
                        lockedFocusDistance = currentDistance
                        focusLocked = true
                        runOnUiThread {
                            focusOverlay.showFocusLocked()
                            val d = if (lockedFocusDistance > 0) String.format(Locale.US, "%.2f diopters", lockedFocusDistance) else "∞"
                            focusStatusText.text      = "✅ Locked ($d). Ready."
                            lockFocusButton.text      = "UNLOCK FOCUS"
                            lockFocusButton.isEnabled = true
                            updateContinueButton()
                        }
                    } else {
                        runOnUiThread {
                            focusOverlay.showFocusFailed()
                            focusStatusText.text      = "⚠️ Focus Failed. Try again."
                            lockFocusButton.text      = "RETRY FOCUS"
                            lockFocusButton.isEnabled = true
                            updateContinueButton()
                        }
                    }
                    updatePreviewRequest()
                }
            }
        }
    }

    private fun triggerAFLock() {
        val session = cameraCaptureSession ?: return
        val builder = previewBuilder ?: return

        runOnUiThread {
            focusStatusText.text = "Focus: Sweeping..."
            lockFocusButton.isEnabled = false
            focusOverlay.showFocusing()
        }

        isFocusing = true

        timeoutRunnable = Runnable {
            if (isFocusing) {
                isFocusing = false
                runOnUiThread {
                    focusOverlay.showFocusFailed()
                    focusStatusText.text = "⚠️ Focus Timed Out."
                    lockFocusButton.text = "RETRY FOCUS"
                    lockFocusButton.isEnabled = true
                    updateContinueButton()
                }
                updatePreviewRequest()
            }
        }
        backgroundHandler?.postDelayed(timeoutRunnable!!, 4000)

        try {
            // STEP 1: The "Wakeup" Tap
            builder.set(CaptureRequest.CONTROL_AF_MODE, CameraMetadata.CONTROL_AF_MODE_AUTO)
            builder.set(CaptureRequest.CONTROL_AF_TRIGGER, CameraMetadata.CONTROL_AF_TRIGGER_START)
            session.capture(builder.build(), null, backgroundHandler)

            // STEP 2: Wait 200ms, then Cancel the Wakeup
            backgroundHandler?.postDelayed({
                if (!isFocusing) return@postDelayed
                try {
                    builder.set(CaptureRequest.CONTROL_AF_TRIGGER, CameraMetadata.CONTROL_AF_TRIGGER_CANCEL)
                    session.capture(builder.build(), null, backgroundHandler)

                    // STEP 3: Wait 100ms, then start the REAL Sweep
                    backgroundHandler?.postDelayed({
                        if (!isFocusing) return@postDelayed
                        try {
                            builder.set(CaptureRequest.CONTROL_AF_TRIGGER, CameraMetadata.CONTROL_AF_TRIGGER_START)
                            session.capture(builder.build(), previewCallback, backgroundHandler)

                            builder.set(CaptureRequest.CONTROL_AF_TRIGGER, CameraMetadata.CONTROL_AF_TRIGGER_IDLE)
                            session.setRepeatingRequest(builder.build(), previewCallback, backgroundHandler)
                        } catch (_: Exception) {}
                    }, 100)
                } catch (_: Exception) {}
            }, 200)
        } catch (_: Exception) {}
    }

    private fun applyLockedSettings(builder: CaptureRequest.Builder) {
        builder.set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_AUTO)
        builder.set(CaptureRequest.CONTROL_AWB_MODE, wbModeValues[currentWbIdx])
        builder.set(CaptureRequest.CONTROL_AE_MODE, CameraMetadata.CONTROL_AE_MODE_OFF)
        builder.set(CaptureRequest.SENSOR_SENSITIVITY, isoValues[currentIsoIdx])
        builder.set(CaptureRequest.SENSOR_EXPOSURE_TIME, shutterNsValues[currentShutterIdx])

        // THE FIX: Turn OIS OFF globally to prevent lens dropping/twitching on a rigid mount.
        builder.set(CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE, CameraMetadata.LENS_OPTICAL_STABILIZATION_MODE_OFF)

        if (isCapturePhase) {
            builder.set(CaptureRequest.CONTROL_AF_TRIGGER, CameraMetadata.CONTROL_AF_TRIGGER_IDLE)
            builder.set(CaptureRequest.CONTROL_AE_PRECAPTURE_TRIGGER, CameraMetadata.CONTROL_AE_PRECAPTURE_TRIGGER_IDLE)
            builder.set(CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE, CameraMetadata.CONTROL_VIDEO_STABILIZATION_MODE_OFF)
            builder.set(CaptureRequest.NOISE_REDUCTION_MODE, CameraMetadata.NOISE_REDUCTION_MODE_OFF)
            builder.set(CaptureRequest.EDGE_MODE, CameraMetadata.EDGE_MODE_OFF)

            if (isManualFocusMode) {
                builder.set(CaptureRequest.CONTROL_AF_MODE, CameraMetadata.CONTROL_AF_MODE_OFF)
                builder.set(CaptureRequest.LENS_FOCUS_DISTANCE, manualFocusDistance)
            } else {
                // Samsung Macro Fix preserved: Keep in AUTO to prevent float quantization error
                builder.set(CaptureRequest.CONTROL_AF_MODE, CameraMetadata.CONTROL_AF_MODE_AUTO)
            }
        } else {
            // Setup Phase
            if (isManualFocusMode) {
                builder.set(CaptureRequest.CONTROL_AF_MODE, CameraMetadata.CONTROL_AF_MODE_OFF)
                builder.set(CaptureRequest.LENS_FOCUS_DISTANCE, manualFocusDistance)
            } else {
                builder.set(CaptureRequest.CONTROL_AF_MODE, CameraMetadata.CONTROL_AF_MODE_AUTO)
            }
        }
    }

    private fun updatePreviewRequest() {
        val session = cameraCaptureSession ?: return
        val builder = previewBuilder ?: return

        if (isFocusing) return
        applyLockedSettings(builder)

        try {
            builder.set(CaptureRequest.CONTROL_AF_TRIGGER, CameraMetadata.CONTROL_AF_TRIGGER_IDLE)
            session.setRepeatingRequest(builder.build(), previewCallback, backgroundHandler)
        } catch (_: Exception) {}
    }

    private fun updateInfoText() {
        runOnUiThread {
            val dist = if (isManualFocusMode) manualFocusDistance else lockedFocusDistance
            val fd = if (dist > 0f) String.format(Locale.US, "%.2f", dist) else "∞"
            val previewTag = if (!isCapturePhase && isPreviewLightOn) " | 💡 L1 Preview" else ""
            cameraInfoText.text = String.format(Locale.US,
                "ISO: %d | SS: %s | WB: %s | FD: %s%s",
                isoValues[currentIsoIdx], shutterLabels[currentShutterIdx],
                wbLabels[currentWbIdx], fd, previewTag)
        }
    }

    // ── Generic Helpers ───────────────────────────────────────────────
    private fun sendToJetson(message: String) {
        webSocket?.send(message)
    }

    private fun sendPreviewLightOn() {
        if (!isJetsonConnected) return
        isPreviewLightOn = true
        sendToJetson("PREVIEW_LIGHT_ON")   // Jetson HTTP-GETs ESP32 /light_on?id=1
        updateInfoText()
    }

    private fun sendPreviewLightOff() {
        if (!isJetsonConnected) return
        isPreviewLightOn = false
        sendToJetson("PREVIEW_LIGHT_OFF")  // Jetson HTTP-GETs ESP32 /all_off
        updateInfoText()
    }

    private fun updateStatus(message: String) {
        runOnUiThread { statusText.text = message }
    }

    // ── Jetson WebSocket Connection ────────────────────────────────────
    private fun connectToJetson() {
        runOnUiThread {
            connectionStatus.text = "CONNECTING TO JETSON..."
            connectionStatus.setBackgroundColor(android.graphics.Color.parseColor("#FF9800"))
        }
        val client = OkHttpClient.Builder()
            .connectTimeout(5, java.util.concurrent.TimeUnit.SECONDS)
            //.pingInterval(25, java.util.concurrent.TimeUnit.SECONDS) // <--- ADDS A 25-SECOND HEARTBEAT
            .build()
        val request = Request.Builder().url(JETSON_WS_URL).build()

        webSocket = client.newWebSocket(request, object : WebSocketListener() {
            override fun onOpen(ws: WebSocket, response: Response) {
                isJetsonConnected = true
                wsRetryHandler.removeCallbacksAndMessages(null) // cancel any pending retries
                runOnUiThread {
                    connectionStatus.text = "CONNECTED TO JETSON"
                    connectionStatus.setBackgroundColor(android.graphics.Color.parseColor("#4CAF50"))
                    updateContinueButton()
                }

            }
            override fun onMessage(ws: WebSocket, text: String) {
                when {
                    text.startsWith("TRIGGER_CAPTURE_") -> {
                        val lightNum = text.removePrefix("TRIGGER_CAPTURE_").toIntOrNull() ?: return
                        currentLight   = lightNum
                        captureCount   = 0
                        isCapturing    = false
                        lightStartTime = System.currentTimeMillis()
                        updateStatus("💡 Light $lightNum ON\n⏳ Capturing...")
                        backgroundHandler?.postDelayed({ captureNext() }, CAPTURE_DELAY_MS)
                    }
                    text == "SEQUENCE_COMPLETE" -> {
                        updateStatus("🎉 Sequence complete!")
                        runOnUiThread {
                            startSequenceButton.visibility = View.GONE
                            uploadButton.visibility  = View.VISIBLE
                            uploadButton.isEnabled = true
                        }
                    }
                    else -> updateStatus("📨 Jetson: $text")
                }
            }
            override fun onClosed(ws: WebSocket, code: Int, reason: String) {
                isJetsonConnected = false
                runOnUiThread {
                    connectionStatus.text = "NOT CONNECTED TO JETSON"
                    connectionStatus.setBackgroundColor(android.graphics.Color.parseColor("#F44336"))
                    updateContinueButton()
                }
            }
            override fun onFailure(ws: WebSocket, t: Throwable, response: Response?) {
                isJetsonConnected = false
                val reason = when {
                    t.message?.contains("refused", ignoreCase = true) == true ->
                        "Server not running on Jetson"
                    t.message?.contains("timeout", ignoreCase = true) == true ->
                        "Connection timed out"
                    else -> t.message ?: "Unknown error"
                }
                android.util.Log.e("JETSON_WS", "WebSocket failure: $reason", t)
                runOnUiThread {
                    connectionStatus.text = "NOT CONNECTED — $reason. Retrying..."
                    connectionStatus.setBackgroundColor(android.graphics.Color.parseColor("#F44336"))
                    updateContinueButton()
                }
                wsRetryHandler.postDelayed({ connectToJetson() }, 3000)
            }
        })
    }

    private fun startCaptureSequence() {

        // --- NEW DATA HYGIENE: Wipe old files before starting new sequence ---
        val dir = File(saveFolder)
        if (dir.exists()) {
            dir.listFiles()?.forEach { it.deleteRecursively() }
        } else {
            dir.mkdirs()
        }



        val framesStr = framesInput.text.toString()
        framesPerLight = framesStr.toIntOrNull() ?: 1
        if (framesPerLight !in 1..10) {
            Toast.makeText(this, "Please enter 1-10 frames", Toast.LENGTH_SHORT).show()
            return
        }
        formatMode = when (formatRadioGroup.checkedRadioButtonId) {
            R.id.formatJpeg -> "JPEG"
            R.id.formatDng  -> "DNG"
            else            -> "BOTH"
        }

        currentLight = 1
        captureCount = 0

        runOnUiThread {
            startSequenceButton.isEnabled = false
            // Force the preview checkboxes off visually
            cbLight1.isChecked = false
            cbLight2.isChecked = false
            cbLight3.isChecked = false
            cbLight4.isChecked = false

            updateStatus("⏳ Settling lights...")
        }

        // Tell the Jetson to turn off all lights immediately
        sendPreviewLightOff()

        // Wait 2.5 seconds (2500 ms) before starting the capture sequence
        Handler(Looper.getMainLooper()).postDelayed({
            updateStatus("📸 Sequence Running...")
            sendToJetson("START")
        }, 2500)
    }

    private fun captureNext() {
        if (isCapturing || captureCount >= framesPerLight) return
        isCapturing = true
        updateStatus("💡 Light $currentLight ON\n📸 Capturing ${captureCount + 1}/$framesPerLight...")
        takePhoto()
    }

    private fun takePhoto() {
        val device = cameraDevice ?: return

        // THE FIX: Use STILL_CAPTURE for quality and disable ZSL to prevent ghost frames
        val builder = device.createCaptureRequest(CameraDevice.TEMPLATE_STILL_CAPTURE).apply {
            if (formatMode == "JPEG" || formatMode == "BOTH") addTarget(jpegReader.surface)
            if (formatMode == "DNG" || formatMode == "BOTH") addTarget(rawReader.surface)

            applyLockedSettings(this)

            // Prevent ghost frames by forcing the sensor to read a fresh exposure
            set(CaptureRequest.CONTROL_ENABLE_ZSL, false)

            val camMgr = getSystemService(Context.CAMERA_SERVICE) as CameraManager
            val chars  = camMgr.getCameraCharacteristics(CAMERA_ID)
            val sensorOri = chars.get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 90

            val displayRotationDegrees = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                display?.rotation ?: Surface.ROTATION_0
            } else {
                @Suppress("DEPRECATION")
                windowManager.defaultDisplay.rotation
            }

            val displayRotationValue = when (displayRotationDegrees) {
                Surface.ROTATION_0 -> 0
                Surface.ROTATION_90 -> 90
                Surface.ROTATION_180 -> 180
                Surface.ROTATION_270 -> 270
                else -> 0
            }
            set(CaptureRequest.JPEG_ORIENTATION, (sensorOri - displayRotationValue + 360) % 360)
        }

        val captureCallback = object : CameraCaptureSession.CaptureCallback() {
            override fun onCaptureCompleted(session: CameraCaptureSession, request: CaptureRequest, result: TotalCaptureResult) {
                pendingCaptureResult = result
                checkAndSave()
            }
        }
        cameraCaptureSession!!.capture(builder.build(), captureCallback, backgroundHandler)
    }

    private fun checkAndSave() {
        val needJpeg = formatMode == "JPEG" || formatMode == "BOTH"
        val needRaw  = formatMode == "DNG"  || formatMode == "BOTH"

        if ((needJpeg && pendingJpegBytes == null) || (needRaw && (pendingRawImage == null || pendingCaptureResult == null))) return

        val lightIdx   = currentLight
        val captureIdx = captureCount + 1

        if (needJpeg) pendingJpegBytes?.let { saveJpeg(it, lightIdx, captureIdx) }
        if (needRaw)  pendingRawImage?.let  { saveDng(it, pendingCaptureResult!!, lightIdx, captureIdx) }

        pendingJpegBytes = null
        pendingRawImage  = null
        pendingCaptureResult = null

        captureCount++
        isCapturing = false

        if (captureCount < framesPerLight) {
            backgroundHandler?.post { captureNext() }
        } else {
            updateStatus("✅ Light $lightIdx saved ($framesPerLight frames)")

            // Wait exactly 1.0 second AFTER capturing before turning the light off
            backgroundHandler?.postDelayed({
                sendToJetson("DONE_$lightIdx")
            }, 1000L)
        }
    }

    private fun saveJpeg(bytes: ByteArray, lightIdx: Int, captureIdx: Int) {
        val filename = String.format(Locale.US, "light_%03d_%d.jpg", lightIdx, captureIdx)
        val dir = File(saveFolder)
        if (!dir.exists()) dir.mkdirs()
        val file = File(dir, filename)
        if (file.exists()) file.delete()
        FileOutputStream(file).use { it.write(bytes) }
    }

    private fun saveDng(image: Image, captureResult: TotalCaptureResult, lightIdx: Int, captureIdx: Int) {
        val filename = String.format(Locale.US, "light_%03d_%d.dng", lightIdx, captureIdx)
        val dir = File(saveFolder)
        if (!dir.exists()) dir.mkdirs()
        val file = File(dir, filename)
        if (file.exists()) file.delete()

        val camMgr = getSystemService(Context.CAMERA_SERVICE) as CameraManager
        val chars = camMgr.getCameraCharacteristics(CAMERA_ID)
        FileOutputStream(file).use { stream ->
            val dngCreator = DngCreator(chars, captureResult)
            dngCreator.writeImage(stream, image)
            dngCreator.close()
        }
        image.close()
    }

    // ── Camera Initialization ─────────────────────────────────────────
    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == CAMERA_REQUEST_CODE && grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        }
    }

    private fun startCamera() {
        backgroundThread = HandlerThread("CameraThread").apply { start() }
        backgroundHandler = Handler(backgroundThread!!.looper)

        val camMgr = getSystemService(Context.CAMERA_SERVICE) as CameraManager
        val chars = camMgr.getCameraCharacteristics(CAMERA_ID)

        try {
            minHardwareFocusDistance = chars.get(CameraCharacteristics.LENS_INFO_MINIMUM_FOCUS_DISTANCE) ?: 0.0f
            maxObservedDiopter = minHardwareFocusDistance
        } catch (_: Exception) {}

        val map = chars.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
        val rawSizes = map?.getOutputSizes(ImageFormat.RAW_SENSOR)
        if (!rawSizes.isNullOrEmpty()) {
            rawWidth = rawSizes[0].width
            rawHeight = rawSizes[0].height
        }

        jpegReader = ImageReader.newInstance(SENSOR_WIDTH, SENSOR_HEIGHT, ImageFormat.JPEG, 2)
        jpegReader.setOnImageAvailableListener({ reader: ImageReader ->
            val image = reader.acquireLatestImage() ?: return@setOnImageAvailableListener
            val buffer = image.planes[0].buffer
            val bytes = ByteArray(buffer.remaining())
            buffer.get(bytes)
            image.close()
            pendingJpegBytes = bytes
            checkAndSave()
        }, backgroundHandler)

        rawReader = ImageReader.newInstance(rawWidth, rawHeight, ImageFormat.RAW_SENSOR, 2)
        rawReader.setOnImageAvailableListener({ reader: ImageReader ->
            val image = reader.acquireLatestImage() ?: return@setOnImageAvailableListener
            pendingRawImage?.close()
            pendingRawImage = image
            checkAndSave()
        }, backgroundHandler)

        if (textureView.isAvailable) {
            fixPreviewTransform(textureView.width, textureView.height)
            openCamera()
        } else {
            textureView.surfaceTextureListener = object : TextureView.SurfaceTextureListener {
                override fun onSurfaceTextureAvailable(s: SurfaceTexture, w: Int, h: Int) {
                    fixPreviewTransform(w, h)
                    openCamera()
                }
                override fun onSurfaceTextureSizeChanged(s: SurfaceTexture, w: Int, h: Int) {
                    fixPreviewTransform(w, h)
                }
                override fun onSurfaceTextureDestroyed(s: SurfaceTexture) = true
                override fun onSurfaceTextureUpdated(s: SurfaceTexture) {}
            }
        }
    }

    private fun openCamera() {
        val mgr = getSystemService(Context.CAMERA_SERVICE) as CameraManager
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            mgr.openCamera(CAMERA_ID, object : CameraDevice.StateCallback() {
                override fun onOpened(camera: CameraDevice) {
                    cameraDevice = camera
                    showPreview()
                }
                override fun onDisconnected(camera: CameraDevice) { camera.close() }
                override fun onError(camera: CameraDevice, error: Int) { camera.close() }
            }, backgroundHandler)
        }
    }

    private fun showPreview() {
        val surfaceTexture = textureView.surfaceTexture
        val pSurface = Surface(surfaceTexture)
        previewSurface = pSurface

        cameraDevice!!.createCaptureSession(
            SessionConfiguration(
                SessionConfiguration.SESSION_REGULAR,
                listOf(OutputConfiguration(pSurface), OutputConfiguration(jpegReader.surface), OutputConfiguration(rawReader.surface)),
                Executors.newSingleThreadExecutor(),
                object : CameraCaptureSession.StateCallback() {
                    override fun onConfigured(session: CameraCaptureSession) {
                        cameraCaptureSession = session
                        previewBuilder = cameraDevice!!.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW)
                        previewBuilder!!.addTarget(pSurface)
                        updatePreviewRequest()
                        updateInfoText()
                    }
                    override fun onConfigureFailed(session: CameraCaptureSession) {
                        updateStatus("❌ Camera session failed")
                    }
                }
            )
        )
    }

    private fun fixPreviewTransform(viewWidth: Int, viewHeight: Int) {
        if (viewWidth == 0 || viewHeight == 0) return
        val camMgr = getSystemService(Context.CAMERA_SERVICE) as CameraManager
        val chars  = camMgr.getCameraCharacteristics(CAMERA_ID)
        val sensorOrientation = chars.get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 90

        val displayRotationDegrees = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            display?.rotation ?: Surface.ROTATION_0
        } else {
            @Suppress("DEPRECATION")
            windowManager.defaultDisplay.rotation
        }

        val displayRotationValue = when (displayRotationDegrees) {
            Surface.ROTATION_0 -> 0
            Surface.ROTATION_90 -> 90
            Surface.ROTATION_180 -> 180
            Surface.ROTATION_270 -> 270
            else -> 0
        }

        val swapped = (sensorOrientation - displayRotationValue + 360) % 180 == 90
        val bufferWidth = if (swapped) SENSOR_HEIGHT else SENSOR_WIDTH
        val bufferHeight = if (swapped) SENSOR_WIDTH else SENSOR_HEIGHT

        textureView.surfaceTexture?.setDefaultBufferSize(bufferWidth, bufferHeight)

        val matrix = Matrix()
        val viewRect = RectF(0f, 0f, viewWidth.toFloat(), viewHeight.toFloat())
        val bufRect = RectF(0f, 0f, bufferHeight.toFloat(), bufferWidth.toFloat())
        val cx = viewRect.centerX()
        val cy = viewRect.centerY()
        bufRect.offset(cx - bufRect.centerX(), cy - bufRect.centerY())
        matrix.setRectToRect(viewRect, bufRect, Matrix.ScaleToFit.FILL)
        val scale = maxOf(viewWidth.toFloat() / bufferHeight, viewHeight.toFloat() / bufferWidth)
        matrix.postScale(scale, scale, cx, cy)
        runOnUiThread { textureView.setTransform(matrix) }
    }

    private fun sampleHistogramFromPreview() {
        backgroundHandler?.post {
            try {
                val full = textureView.bitmap ?: return@post
                val scaled = full.scale(80, 107, false)
                full.recycle()

                val rHist = IntArray(256)
                val gHist = IntArray(256)
                val bHist = IntArray(256)

                val w = scaled.width
                val h = scaled.height

                for (y in 0 until h) {
                    for (x in 0 until w) {
                        val px = scaled[x, y]
                        rHist[(px shr 16) and 0xFF]++
                        gHist[(px shr 8)  and 0xFF]++
                        bHist[ px         and 0xFF]++
                    }
                }
                scaled.recycle()
                if (!isCapturePhase) focusOverlay.updateHistogram(rHist, gHist, bHist)
            } catch (_: Exception) {}
        }
    }

    // ── C++ Batch Processing ──────────────────────────────────────────
    private fun runBatchProcess() {
        updateStatus("Scanning folder for RAW files...")
        finalPreviewImages.clear()

        val camMgr = getSystemService(Context.CAMERA_SERVICE) as CameraManager
        val chars = camMgr.getCameraCharacteristics(CAMERA_ID)
        val sensorOri = chars.get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 90

        val displayRotationDegrees = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            display?.rotation ?: Surface.ROTATION_0
        } else {
            @Suppress("DEPRECATION")
            windowManager.defaultDisplay.rotation
        }

        val displayRotationValue = when (displayRotationDegrees) {
            Surface.ROTATION_0 -> 0
            Surface.ROTATION_90 -> 90
            Surface.ROTATION_180 -> 180
            Surface.ROTATION_270 -> 270
            else -> 0
        }
        val requiredRotation = (sensorOri - displayRotationValue + 360) % 360

        Thread {
            val targetDirectory = saveFolder
            val dir = File(targetDirectory)
            if (!dir.exists()) {
                updateStatus("Error: Directory not found!")
                return@Thread
            }

            val lightGroups = mutableMapOf<String, MutableList<String>>()
            val regExp = Regex("light_(\\d{3})_(\\d+)\\.(dng|DNG)$", RegexOption.IGNORE_CASE)

            dir.listFiles()?.forEach { file ->
                if (file.isFile) {
                    val match = regExp.find(file.name)
                    if (match != null) {
                        val lightId = match.groupValues[1]
                        lightGroups.getOrPut(lightId) { mutableListOf() }.add(file.absolutePath)
                    }
                }
            }

            if (lightGroups.isEmpty()) {
                updateStatus("No matching files found.")
                runOnUiThread { uploadButton.isEnabled = true }
                return@Thread
            }

            val sortedLightIds = lightGroups.keys.sorted()

            for (lightId in sortedLightIds) {
                val groupPaths = lightGroups[lightId]!!
                updateStatus("Crunching Light $lightId...")

                val outputPath = "$targetDirectory/light_$lightId.png"
                val filePathsArray = groupPaths.toTypedArray()

                val result = processAndDenoise(filePathsArray, filePathsArray.size, outputPath, requiredRotation)
                if (result == 1) {
                    finalPreviewImages.add(outputPath)
                }
            }

            updateStatus("Processing Complete! ${finalPreviewImages.size} PNGs ready for Cloud.")

            runOnUiThread {
                textureView.visibility = View.GONE
                previewScrollView.visibility = View.VISIBLE
                previewContainer.removeAllViews()

                for (path in finalPreviewImages) {
                    val bitmap = BitmapFactory.decodeFile(path)
                    val imageView = ImageView(this@MainActivity).apply {
                        layoutParams = LinearLayout.LayoutParams(300, 300).apply { setMargins(8, 0, 8, 0) }
                        scaleType = ImageView.ScaleType.CENTER_CROP
                        setImageBitmap(bitmap)
                        setOnClickListener {
                            val dialog = AlertDialog.Builder(this@MainActivity, android.R.style.Theme_Black_NoTitleBar_Fullscreen).create()
                            val fullImage = ImageView(this@MainActivity).apply {
                                setImageBitmap(bitmap)
                                scaleType = ImageView.ScaleType.FIT_CENTER
                                setOnClickListener { dialog.dismiss() }
                            }
                            dialog.setView(fullImage)
                            dialog.show()
                        }
                    }
                    previewContainer.addView(imageView)
                }

                uploadButton.text = "PROCEED"
                uploadButton.isEnabled = true
                uploadButton.setBackgroundColor(android.graphics.Color.parseColor("#2196F3"))

                uploadButton.setOnClickListener { showUploadSettingsDialog() }
            }
        }.start()
    }

    // ── JSON Payload Generation ───────────────────────────────────────
    private fun generateMetadataJsons(
        zDistance: Double, camDistance: Double, runSurface: Boolean,
        reqMask: Boolean, reqSurface: Boolean, reqDepth: Boolean, reqNormal: Boolean
    ) {
        if (finalPreviewImages.isEmpty()) {
            updateStatus("❌ Error: No PNGs available to upload.")
            return
        }

        val camMgr = getSystemService(Context.CAMERA_SERVICE) as CameraManager
        val chars = camMgr.getCameraCharacteristics(CAMERA_ID)

        val focalLengths = chars.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)
        val focalLength = if (focalLengths != null && focalLengths.isNotEmpty()) focalLengths[0] else 0f
        val sensorSize = chars.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)

        var physWidth = sensorSize?.width?.toDouble() ?: 0.0
        var physHeight = sensorSize?.height?.toDouble() ?: 0.0

        if (rawHeight > rawWidth && physWidth > physHeight) {
            val temp = physWidth; physWidth = physHeight; physHeight = temp
        }

        val dir = File(saveFolder)

        val maskJson = JSONObject().apply {
            put("pipeline_type", "background_removal")
            put("image_paths", JSONArray(finalPreviewImages.map { File(it).name }))
            val worldCoordinates = JSONArray().apply { put(0.0); put(0.0); put(zDistance) }
            put("world_coordinates", worldCoordinates)
            put("generate_csv", runSurface)
            put("return_mask_image", reqMask)
        }
        FileWriter(File(dir, "masking_meta.json")).use { it.write(maskJson.toString(4)) }

        if (runSurface) {
            val reconJson = JSONObject().apply {
                put("pipeline_type", "photometric_stereo")
                put("images", JSONArray(finalPreviewImages.map { File(it).name }))

                val options = BitmapFactory.Options().apply { inJustDecodeBounds = true }
                BitmapFactory.decodeFile(finalPreviewImages[0], options)

                val cameraNode = JSONObject().apply {
                    put("iso", isoValues[currentIsoIdx])
                    put("shutter_speed", shutterLabels[currentShutterIdx])
                    put("focal_length_mm", focalLength.toDouble())
                    put("sensor_width_mm", physWidth)
                    put("sensor_height_mm", physHeight)
                    put("image_width", options.outWidth)
                    put("image_height", options.outHeight)
                    put("object_distance_m", camDistance)
                }
                put("camera_settings", cameraNode)

                val outputsNode = JSONObject().apply {
                    put("3d_surface_html", reqSurface)
                    put("depth_map_png", reqDepth)
                    put("normal_map_png", reqNormal)
                }
                put("requested_outputs", outputsNode)
            }
            FileWriter(File(dir, "reconstruction_meta.json")).use { it.write(reconJson.toString(4)) }
        } else {
            val reconFile = File(dir, "reconstruction_meta.json")
            if (reconFile.exists()) reconFile.delete()
        }

        updateStatus("✅ JSON settings saved! Ready for Cloud.")
    }

    // ── Cloud Options Popup ───────────────────────────────────────────
    private fun showUploadSettingsDialog() {
        val layout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(60, 40, 60, 20)
        }

        val distanceInput = EditText(this).apply { inputType = InputType.TYPE_CLASS_NUMBER or InputType.TYPE_NUMBER_FLAG_DECIMAL; setText("0.019") }
        val camDistInput = EditText(this).apply { inputType = InputType.TYPE_CLASS_NUMBER or InputType.TYPE_NUMBER_FLAG_DECIMAL; setText("0.15") }
        val runSurfaceCheckbox = CheckBox(this).apply { text = "Run 3D Surface Pipeline (Cloud)"; isChecked = true; setPadding(0, 30, 0, 10) }

        val outMaskCb   = CheckBox(this).apply { text = "Masked Image (.png)"; isChecked = true }
        val outSurfaceCb= CheckBox(this).apply { text = "3D Surface (.html)"; isChecked = true }
        val outDepthCb  = CheckBox(this).apply { text = "Depth Map (.png)"; isChecked = true }
        val outNormalCb = CheckBox(this).apply { text = "Normal Map (.png)"; isChecked = true }

        layout.addView(TextView(this).apply { text = "Object thickness from base (Z-axis in meters):"; setPadding(0, 0, 0, 10) })
        layout.addView(distanceInput)
        layout.addView(TextView(this).apply { text = "Camera lens to object distance (meters):"; setPadding(0, 30, 0, 10) })
        layout.addView(camDistInput)
        layout.addView(runSurfaceCheckbox)
        layout.addView(TextView(this).apply { text = "Files to receive back from server:"; setPadding(0, 20, 0, 10); setTypeface(null, Typeface.BOLD) })
        layout.addView(outMaskCb); layout.addView(outSurfaceCb); layout.addView(outDepthCb); layout.addView(outNormalCb)

        val dialog = AlertDialog.Builder(this)
            .setTitle("Cloud Processing Options")
            .setView(layout)
            .setCancelable(false)
            .setPositiveButton("DONE") { _, _ ->
                val zDist = distanceInput.text.toString().toDoubleOrNull() ?: 0.019
                val camDist = camDistInput.text.toString().toDoubleOrNull() ?: 0.15
                generateMetadataJsons(zDist, camDist, runSurfaceCheckbox.isChecked, outMaskCb.isChecked, outSurfaceCb.isChecked, outDepthCb.isChecked, outNormalCb.isChecked)

                uploadButton.text = "SEND FILES TO JETSON"
                uploadButton.setBackgroundColor(android.graphics.Color.parseColor("#4CAF50"))
                uploadButton.setOnClickListener {
                    uploadButton.isEnabled = false
                    uploadToJetson()
                }
            }
            .setNegativeButton("CANCEL", null)
            .create()

        // ── Dynamic Button Logic ──
        fun updateDoneButtonState() {
            val isAnyChecked = outMaskCb.isChecked || outSurfaceCb.isChecked || outDepthCb.isChecked || outNormalCb.isChecked
            dialog.getButton(AlertDialog.BUTTON_POSITIVE)?.isEnabled = isAnyChecked
        }

        val checkListener = CompoundButton.OnCheckedChangeListener { _, _ -> updateDoneButtonState() }
        outMaskCb.setOnCheckedChangeListener(checkListener)
        outSurfaceCb.setOnCheckedChangeListener(checkListener)
        outDepthCb.setOnCheckedChangeListener(checkListener)
        outNormalCb.setOnCheckedChangeListener(checkListener)

        runSurfaceCheckbox.setOnCheckedChangeListener { _, isChecked ->
            outSurfaceCb.isEnabled = isChecked
            outDepthCb.isEnabled = isChecked
            outNormalCb.isEnabled = isChecked
            if (!isChecked) {
                outSurfaceCb.isChecked = false
                outDepthCb.isChecked = false
                outNormalCb.isChecked = false
            } else {
                outSurfaceCb.isChecked = true
                outDepthCb.isChecked = true
                outNormalCb.isChecked = true
            }
            updateDoneButtonState()
        }

        // We must set the initial button state only AFTER the dialog is shown on screen
        dialog.setOnShowListener {
            updateDoneButtonState()
        }

        dialog.show()
    }

    // ── HTTP Upload → poll /status → GET /result ─────────────────────
    private fun uploadToJetson() {
        updateStatus("⬆️ Uploading to Jetson...")
        cancelNetworkTask = false

        Thread {
            try {
                val dir = File(saveFolder)
                val multipart = MultipartBody.Builder().setType(MultipartBody.FORM)
                for (path in finalPreviewImages) {
                    multipart.addFormDataPart(
                        "images", File(path).name,
                        File(path).asRequestBody("image/png".toMediaType())
                    )
                }
                listOf("masking_meta.json", "reconstruction_meta.json").forEach { name ->
                    val f = File(dir, name)
                    if (f.exists()) multipart.addFormDataPart(
                        "metadata", f.name, f.asRequestBody("application/json".toMediaType())
                    )
                }

                // Step 1: Upload files — Execute and track the call
                val uploadClient = OkHttpClient.Builder()
                    .writeTimeout(180, java.util.concurrent.TimeUnit.SECONDS)
                    .readTimeout(30,   java.util.concurrent.TimeUnit.SECONDS)
                    .build()

                val request = Request.Builder().url(JETSON_UPLOAD_URL).post(multipart.build()).build()
                currentOkHttpCall = uploadClient.newCall(request)

                val uploadResp = try {
                    currentOkHttpCall!!.execute()
                } catch (e: Exception) {
                    if (cancelNetworkTask) return@Thread // We killed it intentionally via Retake

                    updateStatus("❌ Upload failed: ${e.message}")
                    runOnUiThread {
                        // Clean reset so the user can just tap it to try again
                        uploadButton.text = "SEND FILES TO JETSON"
                        uploadButton.setBackgroundColor(android.graphics.Color.parseColor("#4CAF50"))
                        uploadButton.isEnabled = true
                    }
                    return@Thread
                }

                val uploadBody = uploadResp.body?.string() ?: ""
                val uploadCode = uploadResp.code
                uploadResp.close()

                if (!uploadResp.isSuccessful) {
                    updateStatus("❌ Upload rejected (HTTP $uploadCode): $uploadBody")
                    runOnUiThread {
                        uploadButton.text = "SEND FILES TO JETSON"
                        uploadButton.setBackgroundColor(android.graphics.Color.parseColor("#4CAF50"))
                        uploadButton.isEnabled = true
                    }
                    return@Thread
                }
                updateStatus("✅ Files received by Jetson ⏳ Processing...")

                // Step 2: Poll /status every 5s until pipeline finishes
                val pollClient = OkHttpClient.Builder()
                    .connectTimeout(10, java.util.concurrent.TimeUnit.SECONDS)
                    .readTimeout(10,   java.util.concurrent.TimeUnit.SECONDS)
                    .build()
                var elapsed = 0

                while (true) {
                    if (cancelNetworkTask) return@Thread // Kill switch

                    Thread.sleep(5000)
                    elapsed += 5

                    if (cancelNetworkTask) return@Thread // Kill switch

                    val statusResp = try {
                        pollClient.newCall(
                            Request.Builder().url(JETSON_STATUS_URL).get().build()
                        ).execute()
                    } catch (e: Exception) {
                        updateStatus("⏳ Processing... ${elapsed}s (${e.message})")
                        continue
                    }
                    val statusJson = try {
                        JSONObject(statusResp.body?.string() ?: "{}")
                    } catch (_: Exception) { statusResp.close(); continue }
                    statusResp.close()

                    val status = statusJson.optString("status", "")
                    val stage  = statusJson.optString("stage",  "")
                    when (status) {
                        "processing" -> updateStatus("⏳ Jetson: $stage... (${elapsed}s)")
                        "done"       -> {
                            // Change to Resend / Fetch mode!
                            runOnUiThread {
                                uploadButton.text = "FETCH RESULTS"
                                uploadButton.setBackgroundColor(android.graphics.Color.parseColor("#2196F3")) // Blue
                                uploadButton.setOnClickListener {
                                    uploadButton.isEnabled = false
                                    downloadResult() // Only triggers the download
                                }
                            }
                            downloadResult()
                            return@Thread
                        }
                        "error"      -> {
                            updateStatus("❌ Pipeline error: ${statusJson.optString("error", "")}")
                            runOnUiThread {
                                uploadButton.text = "SEND FILES TO JETSON"
                                uploadButton.setBackgroundColor(android.graphics.Color.parseColor("#4CAF50"))
                                uploadButton.isEnabled = true
                            }
                            return@Thread
                        }
                        else -> updateStatus("⏳ Starting pipeline... (${elapsed}s)")
                    }
                    if (elapsed > 3600) {
                        updateStatus("❌ Timeout: pipeline >1 hour")
                        runOnUiThread { uploadButton.isEnabled = true }
                        return@Thread
                    }
                }
            } catch (e: Exception) {
                if (cancelNetworkTask) return@Thread
                updateStatus("❌ System error: ${e.message}")
                runOnUiThread {
                    uploadButton.text = "SEND FILES TO JETSON"
                    uploadButton.setBackgroundColor(android.graphics.Color.parseColor("#4CAF50"))
                    uploadButton.isEnabled = true
                }
            }
        }.start()
    }



    private fun extractZipToUserFolder(treeUri: android.net.Uri) {
        Thread {
            try {
                updateStatus("📦 Saving files to your chosen folder...")
                val resolver = contentResolver
                val pickedDir = DocumentFile.fromTreeUri(this, treeUri)
                    ?: throw Exception("Could not access selected folder")

                val zipFile = File(saveFolder, "results.zip")

                ZipInputStream(FileInputStream(zipFile)).use { zis ->
                    var entry = zis.nextEntry
                    while (entry != null) {
                        if (!entry.isDirectory) {
                            // Determine mime type (rough guess based on extension)
                            val mimeType = if (entry.name.endsWith(".html")) "text/html" else "image/png"

                            // Create the file in the user's folder
                            val newFile = pickedDir.createFile(mimeType, entry.name)
                            if (newFile != null) {
                                resolver.openOutputStream(newFile.uri)?.use { fos ->
                                    zis.copyTo(fos)
                                }
                            }
                        }
                        zis.closeEntry()
                        entry = zis.nextEntry
                    }
                }

                // Cleanup the temporary zip
                zipFile.delete()

                updateStatus("✅ All final outputs saved successfully!")
                runOnUiThread { uploadButton.isEnabled = true }

            } catch (e: Exception) {
                updateStatus("❌ Save error: ${e.message}")
                runOnUiThread { uploadButton.isEnabled = true }
            }
        }.start()
    }




    // ── Download HTML from /result → save to PS/surface_3d.html ──────
    private fun downloadResult() {
        Thread {
            if (cancelNetworkTask) return@Thread // THE KILL SWITCH
            try {
                updateStatus("📥 Downloading 3D result from Jetson...")
                val client = OkHttpClient.Builder()
                    .connectTimeout(15, java.util.concurrent.TimeUnit.SECONDS)
                    .readTimeout(120,  java.util.concurrent.TimeUnit.SECONDS)
                    .build()
                val resp = client.newCall(
                    Request.Builder().url(JETSON_RESULT_URL).get().build()
                ).execute()

                if (!resp.isSuccessful) {
                    val body = resp.body?.string() ?: ""
                    resp.close()
                    updateStatus("❌ Download failed (HTTP ${resp.code}): $body")
                    runOnUiThread { uploadButton.isEnabled = true }
                    return@Thread
                }
                val htmlBytes = resp.body?.bytes()
                resp.close()

                if (htmlBytes == null || htmlBytes.isEmpty()) {
                    updateStatus("❌ Empty result from Jetson")
                    runOnUiThread { uploadButton.isEnabled = true }
                    return@Thread
                }

                // Save the ZIP file temporarily to the hidden app cache
                val dir = File(saveFolder)
                if (!dir.exists()) dir.mkdirs()
                val zipFile = File(dir, "results.zip")
                zipFile.writeBytes(htmlBytes)
                val sizeKb = htmlBytes.size / 1024

                // Trigger the Android Folder Picker UI
                runOnUiThread {
                    updateStatus("✅ Downloaded ${sizeKb} KB. Choose where to save outputs...")
                    directoryPickerLauncher.launch(null)
                }
            } catch (e: Exception) {
                updateStatus("❌ Download error: ${e.message}")
                runOnUiThread { uploadButton.isEnabled = true }
            }
        }.start()
    }

    override fun onResume() {
        super.onResume()
        if (cameraDevice == null && ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        }
    }

    override fun onPause() {
        super.onPause()
        cameraCaptureSession?.close(); cameraCaptureSession = null
        cameraDevice?.close();         cameraDevice = null
        previewSurface?.release();     previewSurface = null
        backgroundThread?.quitSafely(); backgroundThread = null
    }

    override fun onDestroy() {
        super.onDestroy()
        wsRetryHandler.removeCallbacksAndMessages(null)
        webSocket?.close(1000, "App closed")
    }

    companion object {
        init { System.loadLibrary("ps_engine") }
        private const val CAMERA_ID     = "0"
        private const val SENSOR_WIDTH  = 4032
        private const val SENSOR_HEIGHT = 3024
        private const val HIST_SAMPLE_EVERY = 10
        private const val CAMERA_REQUEST_CODE = 100
        private const val JETSON_WS_URL     = "ws://192.168.10.1:8080/ws"
        private const val JETSON_UPLOAD_URL  = "http://192.168.10.1:8080/upload_session"
        private const val JETSON_STATUS_URL  = "http://192.168.10.1:8080/status"
        private const val JETSON_RESULT_URL  = "http://192.168.10.1:8080/result"
        private const val CAPTURE_DELAY_MS = 1000L // 1.0s delay BEFORE taking photo
        // (We can delete TOTAL_LIGHT_DURATION_MS entirely, we won't need it)
    }

    external fun processAndDenoise(filePaths: Array<String>, numFiles: Int, outputPath: String, rotationDegrees: Int): Int
}