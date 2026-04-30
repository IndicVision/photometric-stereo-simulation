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
import android.net.nsd.NsdManager
import android.net.nsd.NsdServiceInfo
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
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import kotlin.math.abs

@SuppressLint("SetTextI18n")
class MainActivity : AppCompatActivity() {

    // ── Configuration Arrays ───────────────────────────────────────────
    private val isoValues = intArrayOf(50, 100, 200, 400, 800, 1600, 3200)
    private val shutterNsValues = longArrayOf(
        1_000_000L, 2_000_000L, 4_000_000L, 8_000_000L,
        10_000_000L, 16_666_666L, 20_000_000L, 33_333_333L
    )
    private val shutterLabels = arrayOf(
        "1/1000s","1/500s","1/250s","1/125s","1/100s","1/60s","1/50s","1/30s"
    )
    private val wbModeValues = intArrayOf(
        CameraMetadata.CONTROL_AWB_MODE_AUTO,
        CameraMetadata.CONTROL_AWB_MODE_INCANDESCENT,
        CameraMetadata.CONTROL_AWB_MODE_FLUORESCENT,
        CameraMetadata.CONTROL_AWB_MODE_DAYLIGHT,
        CameraMetadata.CONTROL_AWB_MODE_CLOUDY_DAYLIGHT,
        CameraMetadata.CONTROL_AWB_MODE_SHADE
    )
    private val wbLabels = arrayOf("Auto","Incandescent","Fluorescent","Daylight","Cloudy","Shade")

    // ── Internal State ────────────────────────────────────────────────
    private var currentIsoIdx     = 1
    private var currentShutterIdx = 2
    private var currentWbIdx      = 3   // Daylight default
    private var formatMode        = "BOTH"
    private var framesPerLight    = 1

    private var isManualFocusMode        = false
    private var manualFocusDistance      = 0.0f
    private var minHardwareFocusDistance = 0.0f
    private var maxObservedDiopter       = 0.0f

    private var focusLocked         = false
    private var lockedFocusDistance = 0.0f
    private var isFocusing          = false
    private var timeoutRunnable: Runnable? = null
    private var histFrameCount      = 0

    private var savedProbeShutterNs: Long = 0L
    // ── Auto-Exposure (Probe) State ───────────────────────────────────
    private var autoExposureEnabled    = false
    private var isProbeMode            = false
    private var computedShutterNs:     Long? = null
    private var sensorWhiteLevel:      Int   = 4095
    private var sensorExposureTimeMin: Long  = 1_000_000L
    private var sensorExposureTimeMax: Long  = 100_000_000L

    private var savedSessionFolderForProbe  = ""
    private var savedFramesPerLightForProbe = 1
    private var savedFormatModeForProbe     = "BOTH"
    private var probeFolderPath             = ""

    private var isCapturePhase    = false
    private var isJetsonConnected = false

    @Volatile private var cancelNetworkTask = false
    private var currentOkHttpCall: Call? = null

    private var isPreviewLightOn = false

    // ── Ghost-processing kill switch ──────────────────────────────────
    @Volatile private var cancelBatchProcess = false

    // ── Remembered dialog values (persist across retakes) ────────────
    private var lastCameraToBase  = "6.5"
    private var lastPlatformElev  = "0.0"
    private var lastObjectThick   = "0.1"

    // ── Server Discovery ──────────────────────────────────────────────
    @Volatile private var serverIp: String? = null
    private var nsdManager: NsdManager? = null
    private var discoveryListener: NsdManager.DiscoveryListener? = null
    private val discoveryHandler = Handler(Looper.getMainLooper())

    // ── Capture Loop State ────────────────────────────────────────────
    private var currentLight   = 0
    private var captureCount   = 0
    private var isCapturing    = false
    private var lightStartTime = 0L

    private var rawWidth  = 0
    private var rawHeight = 0
    private var pendingJpegBytes:     ByteArray?          = null
    private var pendingRawImage:      Image?              = null
    private var pendingCaptureResult: TotalCaptureResult? = null

    // ── UI Components ─────────────────────────────────────────────────
    private lateinit var connectionStatus: TextView
    private lateinit var textureView:      TextureView
    private lateinit var focusOverlay:     FocusOverlayView
    private lateinit var cameraInfoText:   TextView

    private lateinit var setupUiContainer:       ScrollView
    private lateinit var isoSeekBar:             SeekBar
    private lateinit var isoValueText:           TextView
    private lateinit var shutterSeekBar:         SeekBar
    private lateinit var shutterValueText:        TextView
    private lateinit var wbSpinner:              Spinner
    private lateinit var focusModeSwitch:        SwitchCompat
    private lateinit var manualFocusContainer:   LinearLayout
    private lateinit var manualFocusSeekBar:     SeekBar
    private lateinit var manualFocusValueText:   TextView
    private lateinit var focusStatusText:        TextView
    private lateinit var lockFocusButton:        Button
    private lateinit var formatRadioGroup:       RadioGroup
    private lateinit var framesInput:            EditText
    private lateinit var continueToCaptureButton: Button
    private lateinit var autoExposureSwitch:     SwitchCompat

    private lateinit var captureUiContainer:  LinearLayout
    private lateinit var statusText:          TextView
    private lateinit var startSequenceButton: Button
    private lateinit var uploadButton:        Button
    private lateinit var retakeButton:        Button
    private lateinit var calibrateButton:     Button          // ← NEW
    private lateinit var previewScrollView:   HorizontalScrollView
    private lateinit var previewContainer:    LinearLayout

    private lateinit var cbLight1:         CheckBox
    private lateinit var cbLight2:         CheckBox
    private lateinit var cbLight3:         CheckBox
    private lateinit var cbLight4:         CheckBox
    private lateinit var btnTurnOffPreview: Button
    private lateinit var gyroText:   TextView
    private lateinit var crosshairV: View
    private lateinit var crosshairH: View

    private lateinit var sensorManager: SensorManager
    private var gravitySensor: Sensor? = null
    // ── Camera Core ────────────────────────────────────────────────────
    private var cameraDevice:          CameraDevice?         = null
    private var cameraCaptureSession:  CameraCaptureSession? = null
    private var backgroundThread:      HandlerThread?         = null
    private var backgroundHandler:     Handler?               = null
    private var previewSurface:        Surface?               = null
    private lateinit var jpegReader:   ImageReader
    private lateinit var rawReader:    ImageReader
    private var previewBuilder: CaptureRequest.Builder? = null

    // ── WebSocket & Storage ───────────────────────────────────────────
    private var webSocket: WebSocket? = null
    private val wsRetryHandler = Handler(Looper.getMainLooper())
    private val finalPreviewImages = mutableListOf<String>()
    private val baseFolder by lazy {
        android.os.Environment
            .getExternalStoragePublicDirectory(android.os.Environment.DIRECTORY_DOWNLOADS)
            .absolutePath + "/PS_Scans"
    }
    private var sessionFolder: String = ""

    // ═══════════════════════════════════════════════════════════════════
    // Lifecycle
    // ═══════════════════════════════════════════════════════════════════

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        window.addFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        bindViews()
        discoverServerThenConnect()
        connectionStatus.setOnLongClickListener {
            getSharedPreferences("photostereo", Context.MODE_PRIVATE)
                .edit().remove("server_ip").apply()
            serverIp = null
            Toast.makeText(this, "Server IP cleared. Re-discovering...", Toast.LENGTH_SHORT).show()
            discoverServerThenConnect()
            true
        }
        setupISO()
        setupShutter()
        setupWB()
        setupManualFocusUI()
        setupFocusButton()
        setupPreviewLightingUI()
        setupAutoExposureSwitch()

        continueToCaptureButton.setOnClickListener { enterCapturePhase() }
        startSequenceButton.setOnClickListener     { startCaptureSequence() }
        retakeButton.setOnClickListener            { performFullRetakeReset() }
        calibrateButton.setOnClickListener         { runCalibration() }

        uploadButton.setOnClickListener {
            uploadButton.isEnabled = false
            runBatchProcess()
        }

        // --- THE IF/ELSE STATEMENTS YOU MENTIONED ---
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(
                this, arrayOf(Manifest.permission.CAMERA), CAMERA_REQUEST_CODE)
        } else {
            startCamera()
        }

        if (Build.VERSION.SDK_INT <= Build.VERSION_CODES.P &&
            ActivityCompat.checkSelfPermission(
                this, Manifest.permission.WRITE_EXTERNAL_STORAGE)
            != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(
                this, arrayOf(Manifest.permission.WRITE_EXTERNAL_STORAGE), 101)
        }

        // --- THE NEW SENSOR INITIALIZATION ---
        // Initialize 3D Rotation Vector Sensor for Leveling
        // Initialize Gravity Sensor for Leveling
        sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        gravitySensor = sensorManager.getDefaultSensor(Sensor.TYPE_GRAVITY)
    }
    override fun onResume() {
        super.onResume()
        if (cameraDevice == null &&
            ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        }
        // Start reading the sensor
        gravitySensor?.let {
            sensorManager.registerListener(sensorListener, it, SensorManager.SENSOR_DELAY_UI)
        }
    }

    override fun onPause() {
        super.onPause()
        cameraCaptureSession?.close(); cameraCaptureSession = null
        cameraDevice?.close();         cameraDevice         = null
        previewSurface?.release();     previewSurface       = null
        backgroundThread?.quitSafely(); backgroundThread    = null

        // Stop reading the sensor
        sensorManager.unregisterListener(sensorListener)
    }

    override fun onDestroy() {
        super.onDestroy()
        wsRetryHandler.removeCallbacksAndMessages(null)
        webSocket?.close(1000, "App closed")
        stopNsdDiscovery()
    }

    // ═══════════════════════════════════════════════════════════════════
    // View Binding
    // ═══════════════════════════════════════════════════════════════════

    private fun bindViews() {
        textureView      = findViewById(R.id.textureView)
        connectionStatus = findViewById(R.id.connectionStatus)
        focusOverlay     = findViewById(R.id.focusOverlay)
        cameraInfoText   = findViewById(R.id.cameraInfoText)
        gyroText   = findViewById(R.id.gyroText)
        crosshairV = findViewById(R.id.crosshairV)
        crosshairH = findViewById(R.id.crosshairH)
        setupUiContainer         = findViewById(R.id.setupUiContainer)
        isoSeekBar               = findViewById(R.id.isoSeekBar)
        isoValueText             = findViewById(R.id.isoValueText)
        shutterSeekBar           = findViewById(R.id.shutterSeekBar)
        shutterValueText         = findViewById(R.id.shutterValueText)
        wbSpinner                = findViewById(R.id.wbSpinner)
        focusModeSwitch          = findViewById(R.id.focusModeSwitch)
        manualFocusContainer     = findViewById(R.id.manualFocusContainer)
        manualFocusSeekBar       = findViewById(R.id.manualFocusSeekBar)
        manualFocusValueText     = findViewById(R.id.manualFocusValueText)
        focusStatusText          = findViewById(R.id.focusStatusText)
        lockFocusButton          = findViewById(R.id.lockFocusButton)
        formatRadioGroup         = findViewById(R.id.formatRadioGroup)
        framesInput              = findViewById(R.id.framesInput)
        continueToCaptureButton  = findViewById(R.id.continueToCaptureButton)
        autoExposureSwitch       = findViewById(R.id.autoExposureSwitch)

        cbLight1          = findViewById(R.id.cbLight1)
        cbLight2          = findViewById(R.id.cbLight2)
        cbLight3          = findViewById(R.id.cbLight3)
        cbLight4          = findViewById(R.id.cbLight4)
        btnTurnOffPreview = findViewById(R.id.btnTurnOffPreview)

        captureUiContainer  = findViewById(R.id.captureUiContainer)
        statusText          = findViewById(R.id.statusText)
        startSequenceButton = findViewById(R.id.startSequenceButton)
        uploadButton        = findViewById(R.id.uploadButton)
        retakeButton        = findViewById(R.id.retakeButton)
        calibrateButton     = findViewById(R.id.calibrateButton)     // ← NEW
        previewScrollView   = findViewById(R.id.previewScrollView)
        previewContainer    = findViewById(R.id.previewContainer)

        continueToCaptureButton.isEnabled = false
    }
    private val sensorListener = object : SensorEventListener {
        override fun onSensorChanged(event: SensorEvent?) {
            // THE FIX: Tell Kotlin to abort if null.
            // This safely unwraps 'event' for the rest of the code!
            if (event == null) return

            if (event.sensor.type == Sensor.TYPE_GRAVITY) {
                // Raw gravity vector pulling on the phone
                val gx = event.values[0].toDouble()
                val gy = event.values[1].toDouble()
                val gz = event.values[2].toDouble()

                // Exact math used by Samsung's *#0*# diagnostic menu
                // Perfectly flat face-up = X: 0, Y: 0, Z: 90
                val x_angle = Math.toDegrees(Math.atan2(gx, Math.sqrt(gy * gy + gz * gz)))
                val y_angle = Math.toDegrees(Math.atan2(gy, Math.sqrt(gx * gx + gz * gz)))
                val z_angle = Math.toDegrees(Math.atan2(gz, Math.sqrt(gx * gx + gy * gy)))

                runOnUiThread {
                    gyroText.text = String.format(Locale.US, "X: %.1f° | Y: %.1f° | Z: %.1f°", x_angle, y_angle, z_angle)

                    // Leveling logic: We only care if X and Y are flat to the table (Z will naturally be ~90)
                    if (abs(x_angle) < 1.5 && abs(y_angle) < 1.5) {
                        crosshairV.setBackgroundColor(android.graphics.Color.GREEN)
                        crosshairH.setBackgroundColor(android.graphics.Color.GREEN)
                        gyroText.setTextColor(android.graphics.Color.GREEN)
                    } else {
                        crosshairV.setBackgroundColor(android.graphics.Color.RED)
                        crosshairH.setBackgroundColor(android.graphics.Color.RED)
                        gyroText.setTextColor(android.graphics.Color.WHITE)
                    }
                }
            }
        }
        override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
    }
    // ═══════════════════════════════════════════════════════════════════
    // Server Discovery: mDNS → 5s timeout → manual IP dialog fallback
    // ═══════════════════════════════════════════════════════════════════

    private fun discoverServerThenConnect() {
        wsRetryHandler.removeCallbacksAndMessages(null)
        webSocket?.close(1000, "Re-discovering")
        webSocket = null

        val prefs = getSharedPreferences("photostereo", Context.MODE_PRIVATE)
        val savedIp = prefs.getString("server_ip", null)
        if (!savedIp.isNullOrBlank()) {
            serverIp = savedIp
            connectToServer(savedIp)
            return
        }

        runOnUiThread {
            connectionStatus.text = "🔍 FINDING SERVER..."
            connectionStatus.setBackgroundColor(android.graphics.Color.parseColor("#FF9800"))
        }

        nsdManager = getSystemService(Context.NSD_SERVICE) as NsdManager

        val timeoutRunnable = Runnable {
            stopNsdDiscovery()
            showManualIpDialog()
        }
        discoveryHandler.postDelayed(timeoutRunnable, 5000)

        discoveryListener = object : NsdManager.DiscoveryListener {
            override fun onStartDiscoveryFailed(serviceType: String, errorCode: Int) {
                discoveryHandler.removeCallbacks(timeoutRunnable)
                showManualIpDialog()
            }
            override fun onStopDiscoveryFailed(serviceType: String, errorCode: Int) {}
            override fun onDiscoveryStarted(serviceType: String) {}
            override fun onDiscoveryStopped(serviceType: String) {}
            override fun onServiceLost(serviceInfo: NsdServiceInfo) {}

            override fun onServiceFound(serviceInfo: NsdServiceInfo) {
                if (serviceInfo.serviceName.contains("PhotoStereo", ignoreCase = true)) {
                    nsdManager?.resolveService(serviceInfo, object : NsdManager.ResolveListener {
                        override fun onResolveFailed(si: NsdServiceInfo, errorCode: Int) {}
                        override fun onServiceResolved(si: NsdServiceInfo) {
                            discoveryHandler.removeCallbacks(timeoutRunnable)
                            stopNsdDiscovery()
                            val ip = si.host.hostAddress ?: return
                            getSharedPreferences("photostereo", Context.MODE_PRIVATE)
                                .edit().putString("server_ip", ip).apply()
                            serverIp = ip
                            connectToServer(ip)
                        }
                    })
                }
            }
        }

        try {
            nsdManager?.discoverServices(
                "_http._tcp", NsdManager.PROTOCOL_DNS_SD, discoveryListener)
        } catch (e: Exception) {
            discoveryHandler.removeCallbacks(timeoutRunnable)
            showManualIpDialog()
        }
    }

    private fun stopNsdDiscovery() {
        try { discoveryListener?.let { nsdManager?.stopServiceDiscovery(it) } }
        catch (_: Exception) {}
        discoveryListener = null
    }

    private fun showManualIpDialog() {
        runOnUiThread {
            val prefs = getSharedPreferences("photostereo", Context.MODE_PRIVATE)
            val input = EditText(this).apply {
                inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_URI
                hint = "e.g. 192.168.137.1"
                setPadding(60, 40, 60, 40)
                setText(prefs.getString("server_ip_manual", ""))
            }
            AlertDialog.Builder(this)
                .setTitle("Server Not Found")
                .setMessage(
                    "Could not auto-discover the laptop server.\n\n" +
                    "Enter the server's IP address manually.\n" +
                    "(Check the laptop terminal — it prints the IP on startup)"
                )
                .setView(input)
                .setCancelable(false)
                .setPositiveButton("CONNECT") { _, _ ->
                    val ip = input.text.toString().trim()
                    if (ip.isNotBlank()) {
                        prefs.edit()
                            .putString("server_ip", ip)
                            .putString("server_ip_manual", ip)
                            .apply()
                        serverIp = ip
                        connectToServer(ip)
                    }
                }
                .setNegativeButton("RETRY") { _, _ ->
                    prefs.edit().remove("server_ip").apply()
                    serverIp = null
                    discoverServerThenConnect()
                }
                .show()
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // WebSocket Connection to Laptop Server
    // ═══════════════════════════════════════════════════════════════════

    private fun connectToServer(ip: String) {
        val wsUrl = "ws://$ip:$SERVER_PORT/ws"
        runOnUiThread {
            connectionStatus.text = "CONNECTING TO SERVER..."
            connectionStatus.setBackgroundColor(android.graphics.Color.parseColor("#FF9800"))
        }
        val client = OkHttpClient.Builder()
            .connectTimeout(5, java.util.concurrent.TimeUnit.SECONDS)
            .build()
        val request = Request.Builder().url(wsUrl).build()

        webSocket = client.newWebSocket(request, object : WebSocketListener() {
            override fun onOpen(ws: WebSocket, response: Response) {
                isJetsonConnected = true
                wsRetryHandler.removeCallbacksAndMessages(null)
                runOnUiThread {
                    connectionStatus.text = "CONNECTED TO SERVER ($ip)"
                    connectionStatus.setBackgroundColor(
                        android.graphics.Color.parseColor("#4CAF50"))
                    updateContinueButton()
                }
            }

            override fun onMessage(ws: WebSocket, text: String) {
                when {
                    text.startsWith("TRIGGER_CAPTURE_") -> {
                        val lightNum = text.removePrefix("TRIGGER_CAPTURE_")
                            .toIntOrNull() ?: return
                        currentLight   = lightNum
                        captureCount   = 0
                        isCapturing    = false
                        lightStartTime = System.currentTimeMillis()
                        updateStatus("💡 Light $lightNum ON\n⏳ Capturing...")
                        backgroundHandler?.postDelayed({ captureNext() }, CAPTURE_DELAY_MS)
                    }
                    text == "SEQUENCE_COMPLETE" -> {
                        if (isProbeMode) {
                            onProbeSequenceComplete()
                        } else {
                            updateStatus("🎉 Sequence complete!")
                            runOnUiThread {
                                startSequenceButton.visibility = View.GONE
                                uploadButton.visibility = View.VISIBLE
                                uploadButton.isEnabled  = true
                            }
                        }
                    }
                    else -> updateStatus("📨 Server: $text")
                }
            }

            override fun onClosed(ws: WebSocket, code: Int, reason: String) {
                isJetsonConnected = false
                runOnUiThread {
                    connectionStatus.text = "NOT CONNECTED TO SERVER"
                    connectionStatus.setBackgroundColor(
                        android.graphics.Color.parseColor("#F44336"))
                    updateContinueButton()
                }
            }

            override fun onFailure(ws: WebSocket, t: Throwable, response: Response?) {
                isJetsonConnected = false
                val reason = when {
                    t.message?.contains("refused", ignoreCase = true) == true ->
                        "Server not running on laptop"
                    t.message?.contains("timeout", ignoreCase = true) == true ->
                        "Connection timed out"
                    else -> t.message ?: "Unknown error"
                }
                android.util.Log.e("SERVER_WS", "WebSocket failure: $reason", t)
                runOnUiThread {
                    connectionStatus.text = "NOT CONNECTED — $reason. Retrying..."
                    connectionStatus.setBackgroundColor(
                        android.graphics.Color.parseColor("#F44336"))
                    updateContinueButton()
                }
                wsRetryHandler.postDelayed({ connectToServer(ip) }, 3000)
            }
        })
    }

    // ═══════════════════════════════════════════════════════════════════
    // Continue Button Gate
    // ═══════════════════════════════════════════════════════════════════

    private fun updateContinueButton() {
        val focusReady = isManualFocusMode || focusLocked
        runOnUiThread {
            continueToCaptureButton.isEnabled = isJetsonConnected && focusReady
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // Phase Transitions
    // ═══════════════════════════════════════════════════════════════════

    private fun enterCapturePhase() {
        isCapturePhase = true
        setupUiContainer.visibility = View.GONE
        captureUiContainer.visibility = View.VISIBLE
        startSequenceButton.visibility = View.VISIBLE
        startSequenceButton.isEnabled  = true
        uploadButton.visibility = View.GONE
        updatePreviewRequest()
        updateInfoText()

        if (autoExposureEnabled && computedShutterNs == null) {
            startSequenceButton.text = "RUN PROBE CAPTURE"
        }
    }

    private fun enterSetupPhase() {
        isCapturePhase = false
        captureUiContainer.visibility = View.GONE
        previewScrollView.visibility  = View.GONE
        textureView.visibility        = View.VISIBLE
        setupUiContainer.visibility   = View.VISIBLE
        focusOverlay.visibility =
            if (isManualFocusMode) View.GONE else View.VISIBLE
        updatePreviewRequest()
        updateInfoText()
    }

    // ═══════════════════════════════════════════════════════════════════
    // Full Retake Reset
    // ═══════════════════════════════════════════════════════════════════

    private fun performFullRetakeReset() {
        updateStatus("🧹 Resetting...")

        cancelNetworkTask = true
        cancelBatchProcess = true
        currentOkHttpCall?.cancel()

        finalPreviewImages.clear()
        captureCount = 0
        currentLight = 0
        isCapturing  = false

        computedShutterNs = null
        isProbeMode       = false

        pendingJpegBytes = null
        pendingRawImage?.close()
        pendingRawImage      = null
        pendingCaptureResult = null

        runOnUiThread {
            previewContainer.removeAllViews()
            previewScrollView.visibility = View.GONE
            textureView.visibility       = View.VISIBLE

            uploadButton.visibility    = View.GONE
            calibrateButton.visibility = View.GONE          // ← NEW
            uploadButton.isEnabled  = true
            uploadButton.text       = "PROCESS RAW IMAGES"
            uploadButton.setBackgroundColor(android.graphics.Color.parseColor("#FF9800"))
            uploadButton.setOnClickListener {
                uploadButton.isEnabled = false
                runBatchProcess()
            }

            startSequenceButton.visibility = View.VISIBLE
            startSequenceButton.isEnabled  = true
            startSequenceButton.text       = if (autoExposureEnabled) "RUN PROBE CAPTURE"
                                             else "START CAPTURE"
            statusText.text = "Ready for Capture"

            enterSetupPhase()
        }

        sendPreviewLightOff()
        sessionFolder = ""
    }

    // ═══════════════════════════════════════════════════════════════════
    // Preview Lighting UI
    // ═══════════════════════════════════════════════════════════════════

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
                sendToServer("SET_PREVIEW:${selectedLights.joinToString("")}")
                isPreviewLightOn = true
                updateInfoText()
            }
        }

        checkBoxes.forEach { it.setOnCheckedChangeListener(checkListener) }
        btnTurnOffPreview.setOnClickListener {
            checkBoxes.forEach { it.isChecked = false }
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // Settings UI Setup
    // ═══════════════════════════════════════════════════════════════════

    private fun setupISO() {
        isoSeekBar.max      = isoValues.size - 1
        isoSeekBar.progress = currentIsoIdx
        isoValueText.text   = "ISO ${isoValues[currentIsoIdx]}"
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
        shutterSeekBar.max      = shutterNsValues.size - 1
        shutterSeekBar.progress = currentShutterIdx
        shutterValueText.text   = shutterLabels[currentShutterIdx]
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
        val adapter = object : ArrayAdapter<String>(
            this, android.R.layout.simple_spinner_item, wbLabels) {
            override fun getView(position: Int, convertView: View?,
                                 parent: android.view.ViewGroup): View {
                val view = super.getView(position, convertView, parent) as TextView
                view.setTextColor(android.graphics.Color.parseColor("#00FF00"))
                view.textSize = 14f
                view.gravity  = android.view.Gravity.END
                return view
            }
            override fun getDropDownView(position: Int, convertView: View?,
                                         parent: android.view.ViewGroup): View {
                val view = super.getDropDownView(position, convertView, parent) as TextView
                view.setTextColor(android.graphics.Color.BLACK)
                view.setPadding(30, 30, 30, 30)
                return view
            }
        }
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        wbSpinner.adapter = adapter
        wbSpinner.setSelection(3)
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
                lockFocusButton.text    = getString(R.string.btn_lock_focus)
                focusStatusText.text    = getString(R.string.focus_status_auto)
                updateContinueButton()
                focusOverlay.resetFocus()
            }
            if (!isFocusing) updatePreviewRequest()
        }

        manualFocusSeekBar.setOnSeekBarChangeListener(
            object : SeekBar.OnSeekBarChangeListener {
                override fun onProgressChanged(seekBar: SeekBar?, progress: Int,
                                               fromUser: Boolean) {
                    if (fromUser) {
                        val overdriveLimit = minHardwareFocusDistance * 1.5f
                        manualFocusDistance = (progress / 1000f) * overdriveLimit
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
                focusLocked         = false
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

    // ═══════════════════════════════════════════════════════════════════
    // Camera Callbacks
    // ═══════════════════════════════════════════════════════════════════

    private val previewCallback = object : CameraCaptureSession.CaptureCallback() {
        override fun onCaptureCompleted(
            session: CameraCaptureSession,
            request: CaptureRequest,
            result: TotalCaptureResult
        ) {
            histFrameCount++
            if (histFrameCount % HIST_SAMPLE_EVERY == 0) sampleHistogramFromPreview()

            val currentDistance = result.get(CaptureResult.LENS_FOCUS_DISTANCE) ?: 0.0f
            if (currentDistance > maxObservedDiopter) maxObservedDiopter = currentDistance

            if (isManualFocusMode && !isCapturePhase) {
                runOnUiThread {
                    val d = if (currentDistance <= 0.0f) "∞"
                            else String.format(Locale.US, "%.2f Diopters", currentDistance)
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
                        focusLocked         = true
                        runOnUiThread {
                            focusOverlay.showFocusLocked()
                            val d = if (lockedFocusDistance > 0)
                                String.format(Locale.US, "%.2f diopters", lockedFocusDistance)
                            else "∞"
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
            focusStatusText.text      = "Focus: Sweeping..."
            lockFocusButton.isEnabled = false
            focusOverlay.showFocusing()
        }

        isFocusing = true

        timeoutRunnable = Runnable {
            if (isFocusing) {
                isFocusing = false
                runOnUiThread {
                    focusOverlay.showFocusFailed()
                    focusStatusText.text      = "⚠️ Focus Timed Out."
                    lockFocusButton.text      = "RETRY FOCUS"
                    lockFocusButton.isEnabled = true
                    updateContinueButton()
                }
                updatePreviewRequest()
            }
        }
        backgroundHandler?.postDelayed(timeoutRunnable!!, 4000)

        try {
            builder.set(CaptureRequest.CONTROL_AF_MODE, CameraMetadata.CONTROL_AF_MODE_AUTO)
            builder.set(CaptureRequest.CONTROL_AF_TRIGGER, CameraMetadata.CONTROL_AF_TRIGGER_START)
            session.capture(builder.build(), null, backgroundHandler)

            backgroundHandler?.postDelayed({
                if (!isFocusing) return@postDelayed
                try {
                    builder.set(CaptureRequest.CONTROL_AF_TRIGGER,
                        CameraMetadata.CONTROL_AF_TRIGGER_CANCEL)
                    session.capture(builder.build(), null, backgroundHandler)

                    backgroundHandler?.postDelayed({
                        if (!isFocusing) return@postDelayed
                        try {
                            builder.set(CaptureRequest.CONTROL_AF_TRIGGER,
                                CameraMetadata.CONTROL_AF_TRIGGER_START)
                            session.capture(builder.build(), previewCallback, backgroundHandler)
                            builder.set(CaptureRequest.CONTROL_AF_TRIGGER,
                                CameraMetadata.CONTROL_AF_TRIGGER_IDLE)
                            session.setRepeatingRequest(builder.build(),
                                previewCallback, backgroundHandler)
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

        val shutterToUse = computedShutterNs ?: shutterNsValues[currentShutterIdx]
        builder.set(CaptureRequest.SENSOR_EXPOSURE_TIME, shutterToUse)

        builder.set(CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE,
            CameraMetadata.LENS_OPTICAL_STABILIZATION_MODE_OFF)

        if (isCapturePhase) {
            builder.set(CaptureRequest.CONTROL_AF_TRIGGER,
                CameraMetadata.CONTROL_AF_TRIGGER_IDLE)
            builder.set(CaptureRequest.CONTROL_AE_PRECAPTURE_TRIGGER,
                CameraMetadata.CONTROL_AE_PRECAPTURE_TRIGGER_IDLE)
            builder.set(CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE,
                CameraMetadata.CONTROL_VIDEO_STABILIZATION_MODE_OFF)
            builder.set(CaptureRequest.NOISE_REDUCTION_MODE,
                CameraMetadata.NOISE_REDUCTION_MODE_OFF)
            builder.set(CaptureRequest.EDGE_MODE, CameraMetadata.EDGE_MODE_OFF)
            builder.set(CaptureRequest.SHADING_MODE, CameraMetadata.SHADING_MODE_OFF)
            builder.set(CaptureRequest.STATISTICS_LENS_SHADING_MAP_MODE, CameraMetadata.STATISTICS_LENS_SHADING_MAP_MODE_OFF)
            builder.set(CaptureRequest.COLOR_CORRECTION_ABERRATION_MODE, CameraMetadata.COLOR_CORRECTION_ABERRATION_MODE_OFF)

            if (isManualFocusMode) {
                builder.set(CaptureRequest.CONTROL_AF_MODE,
                    CameraMetadata.CONTROL_AF_MODE_OFF)
                builder.set(CaptureRequest.LENS_FOCUS_DISTANCE, manualFocusDistance)
            } else {
                builder.set(CaptureRequest.CONTROL_AF_MODE,
                    CameraMetadata.CONTROL_AF_MODE_AUTO)
            }
        } else {
            if (isManualFocusMode) {
                builder.set(CaptureRequest.CONTROL_AF_MODE,
                    CameraMetadata.CONTROL_AF_MODE_OFF)
                builder.set(CaptureRequest.LENS_FOCUS_DISTANCE, manualFocusDistance)
            } else {
                builder.set(CaptureRequest.CONTROL_AF_MODE,
                    CameraMetadata.CONTROL_AF_MODE_AUTO)
            }
        }
    }

    private fun updatePreviewRequest() {
        val session = cameraCaptureSession ?: return
        val builder = previewBuilder ?: return
        if (isFocusing) return
        applyLockedSettings(builder)
        try {
            builder.set(CaptureRequest.CONTROL_AF_TRIGGER,
                CameraMetadata.CONTROL_AF_TRIGGER_IDLE)
            session.setRepeatingRequest(builder.build(), previewCallback, backgroundHandler)
        } catch (_: Exception) {}
    }

    private fun updateInfoText() {
        runOnUiThread {
            val dist = if (isManualFocusMode) manualFocusDistance else lockedFocusDistance
            val fd   = if (dist > 0f) String.format(Locale.US, "%.2f", dist) else "∞"
            val previewTag = if (!isCapturePhase && isPreviewLightOn) " | 💡 Preview" else ""
            val ssLabel = computedShutterNs
                ?.let { "AUTO(%.2fms)".format(it / 1_000_000.0) }
                ?: shutterLabels[currentShutterIdx]
            cameraInfoText.text = String.format(Locale.US,
                "ISO: %d | SS: %s | WB: %s | FD: %s%s",
                isoValues[currentIsoIdx], ssLabel,
                wbLabels[currentWbIdx], fd, previewTag)
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // Generic Helpers
    // ═══════════════════════════════════════════════════════════════════

    private fun sendToServer(message: String) { webSocket?.send(message) }

    private fun sendPreviewLightOn() {
        if (!isJetsonConnected) return
        isPreviewLightOn = true
        sendToServer("PREVIEW_LIGHT_ON")
        updateInfoText()
    }

    private fun sendPreviewLightOff() {
        if (!isJetsonConnected) return
        isPreviewLightOn = false
        sendToServer("PREVIEW_LIGHT_OFF")
        updateInfoText()
    }

    private fun updateStatus(message: String) {
        runOnUiThread { statusText.text = message }
    }

    // ═══════════════════════════════════════════════════════════════════
    // Capture Sequence
    // ═══════════════════════════════════════════════════════════════════

    private fun startCaptureSequence() {
        if (autoExposureEnabled && computedShutterNs == null) {
            runProbeSequence()
            return
        }

        val timestamp = java.text.SimpleDateFormat("MMdd_HHmm", Locale.US)
            .format(java.util.Date())
        sessionFolder = "$baseFolder/${timestamp}_ps_scan"
        File(sessionFolder).mkdirs()

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
            cbLight1.isChecked = false
            cbLight2.isChecked = false
            cbLight3.isChecked = false
            cbLight4.isChecked = false
            updateStatus("⏳ Settling lights...")
        }

        sendPreviewLightOff()

        Handler(Looper.getMainLooper()).postDelayed({
            updateStatus("📸 Sequence Running...")
            sendToServer("START")
        }, 2500)
    }

    private fun captureNext() {
        if (isCapturing || captureCount >= framesPerLight) return
        isCapturing = true
        updateStatus("💡 Light $currentLight ON\n" +
                     "📸 Capturing ${captureCount + 1}/$framesPerLight...")
        takePhoto()
    }

    private fun takePhoto() {
        val device = cameraDevice ?: return
        val builder = device.createCaptureRequest(CameraDevice.TEMPLATE_STILL_CAPTURE).apply {
            if (formatMode == "JPEG" || formatMode == "BOTH") addTarget(jpegReader.surface)
            if (formatMode == "DNG"  || formatMode == "BOTH") addTarget(rawReader.surface)

            applyLockedSettings(this)
            set(CaptureRequest.CONTROL_ENABLE_ZSL, false)

            val camMgr = getSystemService(Context.CAMERA_SERVICE) as CameraManager
            val chars  = camMgr.getCameraCharacteristics(CAMERA_ID)
            val sensorOri = chars.get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 90

            val displayRotDeg = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                display?.rotation ?: Surface.ROTATION_0
            } else {
                @Suppress("DEPRECATION") windowManager.defaultDisplay.rotation
            }
            val displayRotVal = when (displayRotDeg) {
                Surface.ROTATION_0   -> 0
                Surface.ROTATION_90  -> 90
                Surface.ROTATION_180 -> 180
                Surface.ROTATION_270 -> 270
                else                 -> 0
            }
            set(CaptureRequest.JPEG_ORIENTATION,
                (sensorOri - displayRotVal + 360) % 360)
        }

        val cb = object : CameraCaptureSession.CaptureCallback() {
            override fun onCaptureCompleted(
                session: CameraCaptureSession,
                request: CaptureRequest,
                result: TotalCaptureResult
            ) {
                pendingCaptureResult = result
                checkAndSave()
            }
        }
        cameraCaptureSession!!.capture(builder.build(), cb, backgroundHandler)
    }

    private fun checkAndSave() {
        val needJpeg = formatMode == "JPEG" || formatMode == "BOTH"
        val needRaw  = formatMode == "DNG"  || formatMode == "BOTH"

        if ((needJpeg && pendingJpegBytes == null) ||
            (needRaw  && (pendingRawImage == null || pendingCaptureResult == null))) {

            if (formatMode == "BOTH") {
                backgroundHandler?.postDelayed({
                    if (isCapturing && (pendingJpegBytes == null || pendingRawImage == null)) {
                        android.util.Log.w("PhotoStereo",
                            "Frame buffer timeout — one buffer never arrived. Skipping frame.")
                        pendingJpegBytes = null
                        pendingRawImage?.close(); pendingRawImage = null
                        pendingCaptureResult = null
                        captureCount++
                        isCapturing = false
                        if (captureCount < framesPerLight) {
                            captureNext()
                        } else {
                            updateStatus("⚠️ Light $currentLight: frame dropped, continuing")
                            backgroundHandler?.postDelayed({ sendToServer("DONE_$currentLight") }, 500L)
                        }
                    }
                }, 5000L)
            }
            return
        }

        val lightIdx   = currentLight
        val captureIdx = captureCount + 1

        if (needJpeg) pendingJpegBytes?.let { saveJpeg(it, lightIdx, captureIdx) }
        if (needRaw)  pendingRawImage?.let  { saveDng(it, pendingCaptureResult!!, lightIdx, captureIdx) }

        pendingJpegBytes     = null
        pendingRawImage      = null
        pendingCaptureResult = null

        captureCount++
        isCapturing = false

        if (captureCount < framesPerLight) {
            backgroundHandler?.post { captureNext() }
        } else {
            updateStatus("✅ Light $lightIdx saved ($framesPerLight frames)")
            backgroundHandler?.postDelayed({
                sendToServer("DONE_$lightIdx")
            }, 1000L)
        }
    }

    private fun saveJpeg(bytes: ByteArray, lightIdx: Int, captureIdx: Int) {
        val filename = String.format(Locale.US, "light_%03d_%d.jpg", lightIdx, captureIdx)
        val dir  = File(sessionFolder).also { if (!it.exists()) it.mkdirs() }
        val file = File(dir, filename).also { if (it.exists()) it.delete() }
        try {
            FileOutputStream(file).use { it.write(bytes) }
        } catch (e: Exception) {
            updateStatus("❌ Save failed (storage full?): ${e.message}")
        }
    }

    private fun saveDng(image: Image, captureResult: TotalCaptureResult,
                        lightIdx: Int, captureIdx: Int) {
        val filename = String.format(Locale.US, "light_%03d_%d.dng", lightIdx, captureIdx)
        val dir  = File(sessionFolder).also { if (!it.exists()) it.mkdirs() }
        val file = File(dir, filename).also { if (it.exists()) it.delete() }
        val camMgr = getSystemService(Context.CAMERA_SERVICE) as CameraManager
        val chars  = camMgr.getCameraCharacteristics(CAMERA_ID)
        try {
            FileOutputStream(file).use { stream ->
                val dng = DngCreator(chars, captureResult)
                dng.writeImage(stream, image)
                dng.close()
            }
        } catch (e: Exception) {
            updateStatus("❌ Save failed (storage full?): ${e.message}")
        } finally {
            image.close()
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // Camera Initialisation
    // ═══════════════════════════════════════════════════════════════════

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<out String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == CAMERA_REQUEST_CODE &&
            grantResults.isNotEmpty() &&
            grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        }
    }

    private fun startCamera() {
        backgroundThread = HandlerThread("CameraThread").apply { start() }
        backgroundHandler = Handler(backgroundThread!!.looper)

        val camMgr = getSystemService(Context.CAMERA_SERVICE) as CameraManager
        val chars  = camMgr.getCameraCharacteristics(CAMERA_ID)
        try {
            minHardwareFocusDistance =
                chars.get(CameraCharacteristics.LENS_INFO_MINIMUM_FOCUS_DISTANCE) ?: 0.0f
            maxObservedDiopter = minHardwareFocusDistance
        } catch (_: Exception) {}

        sensorWhiteLevel = chars.get(CameraCharacteristics.SENSOR_INFO_WHITE_LEVEL) ?: 4095
        val expRange = chars.get(CameraCharacteristics.SENSOR_INFO_EXPOSURE_TIME_RANGE)
        if (expRange != null) {
            sensorExposureTimeMin = expRange.lower
            sensorExposureTimeMax = expRange.upper
        }

        val map      = chars.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
        val rawSizes = map?.getOutputSizes(ImageFormat.RAW_SENSOR)
        if (rawSizes != null && rawSizes.isNotEmpty()) {
            rawWidth  = rawSizes[0].width
            rawHeight = rawSizes[0].height
        }

        jpegReader = ImageReader.newInstance(SENSOR_WIDTH, SENSOR_HEIGHT, ImageFormat.JPEG, 2)
        jpegReader.setOnImageAvailableListener({ reader ->
            val image  = reader.acquireLatestImage() ?: return@setOnImageAvailableListener
            val buffer = image.planes[0].buffer
            val bytes  = ByteArray(buffer.remaining())
            buffer.get(bytes)
            image.close()
            pendingJpegBytes = bytes
            checkAndSave()
        }, backgroundHandler)

        rawReader = ImageReader.newInstance(rawWidth, rawHeight, ImageFormat.RAW_SENSOR, 2)
        rawReader.setOnImageAvailableListener({ reader ->
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
                    fixPreviewTransform(w, h); openCamera()
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
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED) {
            mgr.openCamera(CAMERA_ID, object : CameraDevice.StateCallback() {
                override fun onOpened(camera: CameraDevice) {
                    cameraDevice = camera; showPreview()
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
                listOf(
                    OutputConfiguration(pSurface),
                    OutputConfiguration(jpegReader.surface),
                    OutputConfiguration(rawReader.surface)
                ),
                Executors.newSingleThreadExecutor(),
                object : CameraCaptureSession.StateCallback() {
                    override fun onConfigured(session: CameraCaptureSession) {
                        cameraCaptureSession = session
                        previewBuilder = cameraDevice!!.createCaptureRequest(
                            CameraDevice.TEMPLATE_PREVIEW)
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

        val displayRotDeg = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            display?.rotation ?: Surface.ROTATION_0
        } else {
            @Suppress("DEPRECATION") windowManager.defaultDisplay.rotation
        }
        val displayRotVal = when (displayRotDeg) {
            Surface.ROTATION_0   -> 0
            Surface.ROTATION_90  -> 90
            Surface.ROTATION_180 -> 180
            Surface.ROTATION_270 -> 270
            else                 -> 0
        }

        // Determine if the sensor is rotated 90 degrees relative to the screen (Portrait mode)
        val swapped      = (sensorOrientation - displayRotVal + 360) % 180 == 90
        val bufferWidth  = if (swapped) SENSOR_HEIGHT else SENSOR_WIDTH
        val bufferHeight = if (swapped) SENSOR_WIDTH  else SENSOR_HEIGHT

        textureView.surfaceTexture?.setDefaultBufferSize(bufferWidth, bufferHeight)

        val matrix = Matrix()

        // TextureView inherently stretches the 4:3 buffer to fill the mismatched view bounds.
        // We calculate exactly how much it stretched X and Y.
        val scaleX = viewWidth.toFloat() / bufferWidth.toFloat()
        val scaleY = viewHeight.toFloat() / bufferHeight.toFloat()

        // By taking the minOf, we enforce Letterboxing (adding black bars to preserve 4:3).
        // (If you ever wanted Center-Crop instead, you would change this to maxOf).
        val scale = minOf(scaleX, scaleY)

        // We mathematically "undo" the uneven squish by applying the corrective inverse ratio
        // and centering the matrix perfectly in the middle of the view.
        matrix.setScale(scale / scaleX, scale / scaleY, viewWidth / 2f, viewHeight / 2f)

        runOnUiThread { textureView.setTransform(matrix) }
    }

    private fun sampleHistogramFromPreview() {
        backgroundHandler?.post {
            try {
                val full   = textureView.bitmap ?: return@post
                val scaled = full.scale(80, 107, false)
                full.recycle()

                val rHist = IntArray(256); val gHist = IntArray(256); val bHist = IntArray(256)
                for (y in 0 until scaled.height) {
                    for (x in 0 until scaled.width) {
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

    // ═══════════════════════════════════════════════════════════════════
    // C++ RAW Batch Processing
    // ═══════════════════════════════════════════════════════════════════

    private fun runBatchProcess() {
        cancelBatchProcess = false
        updateStatus("Scanning folder for RAW files...")
        finalPreviewImages.clear()

        val camMgr = getSystemService(Context.CAMERA_SERVICE) as CameraManager
        val chars  = camMgr.getCameraCharacteristics(CAMERA_ID)
        val sensorOri = chars.get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 90

        val displayRotDeg = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            display?.rotation ?: Surface.ROTATION_0
        } else {
            @Suppress("DEPRECATION") windowManager.defaultDisplay.rotation
        }
        val displayRotVal = when (displayRotDeg) {
            Surface.ROTATION_0   -> 0; Surface.ROTATION_90  -> 90
            Surface.ROTATION_180 -> 180; Surface.ROTATION_270 -> 270; else -> 0
        }
        val requiredRotation = (sensorOri - displayRotVal + 360) % 360

        Thread {
            val dir = File(sessionFolder)
            if (!dir.exists()) { updateStatus("Error: Session folder not found!"); return@Thread }

            val lightGroups = mutableMapOf<String, MutableList<String>>()
            val regExp = Regex("light_(\\d{3})_(\\d+)\\.(dng|DNG)$", RegexOption.IGNORE_CASE)

            dir.listFiles()?.forEach { file ->
                if (file.isFile) {
                    regExp.find(file.name)?.let { match ->
                        lightGroups
                            .getOrPut(match.groupValues[1]) { mutableListOf() }
                            .add(file.absolutePath)
                    }
                }
            }

            if (lightGroups.isEmpty()) {
                updateStatus("No matching files found.")
                runOnUiThread { uploadButton.isEnabled = true }
                return@Thread
            }

            for (lightId in lightGroups.keys.sorted()) {
                if (cancelBatchProcess) return@Thread
                val groupPaths  = lightGroups[lightId]!!
                val outputPath  = "$sessionFolder/light_$lightId.png"
                val pathsArray  = groupPaths.toTypedArray()
                val scalingFactor = 65535f / sensorWhiteLevel.toFloat()
                updateStatus("Crunching Light $lightId...")

                val result = processAndDenoise(pathsArray, pathsArray.size, outputPath, requiredRotation, scalingFactor)

                if (result == 1) {
                    finalPreviewImages.add(outputPath)
                }
            }

            if (cancelBatchProcess) return@Thread

            updateStatus("Processing Complete! ${finalPreviewImages.size} PNGs ready.")

            runOnUiThread {
                textureView.visibility        = View.GONE
                previewScrollView.visibility  = View.VISIBLE
                previewContainer.removeAllViews()

                for (path in finalPreviewImages) {
                    val previewPath = path.replace(".png", "_preview.png")
                    val bitmap = BitmapFactory.decodeFile(previewPath)

                    val imageView = ImageView(this@MainActivity).apply {
                        layoutParams = LinearLayout.LayoutParams(300, 300).apply {
                            setMargins(8, 0, 8, 0)
                        }
                        scaleType = ImageView.ScaleType.CENTER_CROP
                        setImageBitmap(bitmap)
                        setOnClickListener {
                            val dialog = AlertDialog.Builder(this@MainActivity,
                                android.R.style.Theme_Black_NoTitleBar_Fullscreen).create()
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

                // Show calibrate button now that 4 PNGs are ready        // ← NEW
                calibrateButton.visibility = View.VISIBLE                 // ← NEW
            }
        }.start()
    }

    // ═══════════════════════════════════════════════════════════════════
    // JSON Metadata Generation
    // ═══════════════════════════════════════════════════════════════════

    private fun generateMetadataJsons(
        zDistance: Double,
        cameraToBase: Double,
        platformElevation: Double,
        objectThickness: Double,
        runSurface: Boolean,
        reqMask: Boolean, reqSurface: Boolean, reqDepth: Boolean, reqNormal: Boolean,
        detrendingMode: String = "none"                                    // ← NEW param
    ) {
        if (finalPreviewImages.isEmpty()) {
            updateStatus("❌ Error: No PNGs available to upload.")
            return
        }

        val camMgr  = getSystemService(Context.CAMERA_SERVICE) as CameraManager
        val chars   = camMgr.getCameraCharacteristics(CAMERA_ID)
        val focalLengths = chars.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)
        val focalLength = if (focalLengths != null && focalLengths.isNotEmpty()) focalLengths[0] else 0f
        val sensorSize   = chars.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)

        var physWidth  = sensorSize?.width?.toDouble()  ?: 0.0
        var physHeight = sensorSize?.height?.toDouble() ?: 0.0
        if (rawHeight > rawWidth && physWidth > physHeight) {
            val tmp = physWidth; physWidth = physHeight; physHeight = tmp
        }

        val dir = File(sessionFolder)

        val maskJson = JSONObject().apply {
            put("pipeline_type", "background_removal")
            put("image_paths", JSONArray(finalPreviewImages.map { File(it).name }))
            put("world_coordinates", JSONArray().apply {
                put(0.0); put(0.0); put(zDistance)
            })
            put("generate_csv",      runSurface)
            put("return_mask_image", reqMask)
        }
        FileWriter(File(dir, "masking_meta.json")).use { it.write(maskJson.toString(4)) }

        if (runSurface) {
            val options = BitmapFactory.Options().apply { inJustDecodeBounds = true }
            BitmapFactory.decodeFile(finalPreviewImages[0], options)

            val shutterLabel = computedShutterNs
                ?.let { "AUTO(%.2fms)".format(it / 1_000_000.0) }
                ?: shutterLabels[currentShutterIdx]

            val reconJson = JSONObject().apply {
                put("pipeline_type", "photometric_stereo")
                put("images", JSONArray(finalPreviewImages.map { File(it).name }))
                put("camera_settings", JSONObject().apply {
                    put("iso",              isoValues[currentIsoIdx])
                    put("shutter_speed",    shutterLabel)
                    put("focal_length_mm",  focalLength.toDouble())
                    put("sensor_width_mm",  physWidth)
                    put("sensor_height_mm", physHeight)
                    put("image_width",      options.outWidth)
                    put("image_height",     options.outHeight)
                    put("object_distance_m", cameraToBase - platformElevation - objectThickness)
                })
                put("camera_to_base_m",     cameraToBase)
                put("platform_elevation_m", platformElevation)
                put("object_thickness_m",   objectThickness)
                put("detrending_mode",      detrendingMode)               // ← NEW
                put("requested_outputs", JSONObject().apply {
                    put("3d_surface_html", reqSurface)
                    put("depth_map_png",   reqDepth)
                    put("normal_map_png",  reqNormal)
                })
            }
            FileWriter(File(dir, "reconstruction_meta.json"))
                .use { it.write(reconJson.toString(4)) }
        } else {
            File(dir, "reconstruction_meta.json").let { if (it.exists()) it.delete() }
        }

        updateStatus("✅ Settings saved! Ready to send to server.")
    }

    // ═══════════════════════════════════════════════════════════════════
    // Upload Settings Dialog
    // ═══════════════════════════════════════════════════════════════════

    private fun showUploadSettingsDialog() {
        val layout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(60, 40, 60, 20)
        }

        val cameraToBaseInput = EditText(this).apply {
            inputType = InputType.TYPE_CLASS_NUMBER or InputType.TYPE_NUMBER_FLAG_DECIMAL
            setText(lastCameraToBase)
        }
        val platformElevInput = EditText(this).apply {
            inputType = InputType.TYPE_CLASS_NUMBER or InputType.TYPE_NUMBER_FLAG_DECIMAL
            setText(lastPlatformElev)
        }
        val objectThickInput = EditText(this).apply {
            inputType = InputType.TYPE_CLASS_NUMBER or InputType.TYPE_NUMBER_FLAG_DECIMAL
            setText(lastObjectThick)
        }

        val runSurfaceCheckbox = CheckBox(this).apply {
            text = "Run 3D Surface Pipeline"; isChecked = true; setPadding(0, 30, 0, 10)
        }
        val outMaskCb    = CheckBox(this).apply { text = "Masked Image (.png)";  isChecked = true }
        val outSurfaceCb = CheckBox(this).apply { text = "3D Surface (.html)";   isChecked = true }
        val outDepthCb   = CheckBox(this).apply { text = "Depth Map (.png)";     isChecked = true }
        val outNormalCb  = CheckBox(this).apply { text = "Normal Map (.png)";    isChecked = true }

        // ── Detrending RadioGroup ─────────────────────────────────────── ← NEW
        val rbNone      = RadioButton(this).apply { text = "None";      id = View.generateViewId() }
        val rbLinear    = RadioButton(this).apply { text = "Linear";    id = View.generateViewId() }
        val rbQuadratic = RadioButton(this).apply { text = "Quadratic"; id = View.generateViewId() }
        val detrendGroup = RadioGroup(this).apply {
            orientation = RadioGroup.HORIZONTAL
            addView(rbNone)
            addView(rbLinear)
            addView(rbQuadratic)
            check(rbNone.id)
        }

        layout.addView(TextView(this).apply {
            text = "Camera to base distance (cm):"; setPadding(0, 0, 0, 6)
        })
        layout.addView(cameraToBaseInput)
        layout.addView(TextView(this).apply {
            text = "Platform elevation (cm):"; setPadding(0, 24, 0, 6)
        })
        layout.addView(platformElevInput)
        layout.addView(TextView(this).apply {
            text = "Object thickness (cm):"; setPadding(0, 24, 0, 6)
        })
        layout.addView(objectThickInput)
        layout.addView(runSurfaceCheckbox)
        layout.addView(TextView(this).apply {
            text = "Files to receive back:"; setPadding(0, 20, 0, 10)
            setTypeface(null, Typeface.BOLD)
        })
        layout.addView(outMaskCb)
        layout.addView(outSurfaceCb)
        layout.addView(outDepthCb)
        layout.addView(outNormalCb)
        // ── Detrending section ─────────────────────────────────────────── ← NEW
        layout.addView(TextView(this).apply {
            text = "Surface detrending:"; setPadding(0, 28, 0, 8)
            setTypeface(null, Typeface.BOLD)
        })
        layout.addView(detrendGroup)

        val dialog = AlertDialog.Builder(this)
            .setTitle("Processing Options")
            .setView(layout)
            .setCancelable(false)
            .setPositiveButton("DONE") { _, _ ->
                val cameraToBase    = (cameraToBaseInput.text.toString().toDoubleOrNull() ?: 6.5)  / 100.0
                val platformElev    = (platformElevInput.text.toString().toDoubleOrNull()  ?: 0.0)  / 100.0
                val objectThickness = (objectThickInput.text.toString().toDoubleOrNull()   ?: 0.1)  / 100.0
                lastCameraToBase = cameraToBaseInput.text.toString()
                lastPlatformElev = platformElevInput.text.toString()
                lastObjectThick  = objectThickInput.text.toString()
                val zDistance      = platformElev + objectThickness
                // ── Read detrending selection ──────────────────────────── ← NEW
                val detrendingMode = when (detrendGroup.checkedRadioButtonId) {
                    rbLinear.id    -> "linear"
                    rbQuadratic.id -> "quadratic"
                    else           -> "none"
                }
                generateMetadataJsons(
                    zDistance, cameraToBase, platformElev, objectThickness,
                    runSurfaceCheckbox.isChecked,
                    outMaskCb.isChecked, outSurfaceCb.isChecked,
                    outDepthCb.isChecked, outNormalCb.isChecked,
                    detrendingMode                                          // ← NEW
                )
                uploadButton.text = "SEND FILES TO SERVER"
                uploadButton.setBackgroundColor(
                    android.graphics.Color.parseColor("#4CAF50"))
                uploadButton.setOnClickListener {
                    uploadButton.isEnabled = false
                    uploadToServer()
                }
            }
            .setNegativeButton("CANCEL", null)
            .create()

        fun updateDoneButtonState() {
            val anyChecked = outMaskCb.isChecked || outSurfaceCb.isChecked ||
                             outDepthCb.isChecked || outNormalCb.isChecked
            dialog.getButton(AlertDialog.BUTTON_POSITIVE)?.isEnabled = anyChecked
        }
        val checkListener = CompoundButton.OnCheckedChangeListener { _, _ ->
            updateDoneButtonState()
        }
        listOf(outMaskCb, outSurfaceCb, outDepthCb, outNormalCb)
            .forEach { it.setOnCheckedChangeListener(checkListener) }

        runSurfaceCheckbox.setOnCheckedChangeListener { _, isChecked ->
            outSurfaceCb.isEnabled  = isChecked; outDepthCb.isEnabled = isChecked
            outNormalCb.isEnabled   = isChecked
            // Detrending only relevant when surface pipeline runs        // ← NEW
            detrendGroup.isEnabled  = isChecked                          // ← NEW
            rbNone.isEnabled        = isChecked                          // ← NEW
            rbLinear.isEnabled      = isChecked                          // ← NEW
            rbQuadratic.isEnabled   = isChecked                          // ← NEW
            if (!isChecked) {
                outSurfaceCb.isChecked = false
                outDepthCb.isChecked   = false
                outNormalCb.isChecked  = false
                detrendGroup.check(rbNone.id)                            // ← NEW
            } else {
                outSurfaceCb.isChecked = true
                outDepthCb.isChecked   = true
                outNormalCb.isChecked  = true
            }
            updateDoneButtonState()
        }
        dialog.setOnShowListener { updateDoneButtonState() }
        dialog.show()
    }

    // ═══════════════════════════════════════════════════════════════════
    // HTTP Upload → poll /status → GET /result
    // ═══════════════════════════════════════════════════════════════════

    private fun uploadToServer() {
        val ip = serverIp
        if (ip == null) {
            updateStatus("❌ Server IP not known. Tap RETRY to re-discover.")
            runOnUiThread { uploadButton.isEnabled = true }
            return
        }

        val uploadUrl = "http://$ip:$SERVER_PORT/upload_session"
        val statusUrl = "http://$ip:$SERVER_PORT/status"

        updateStatus("⬆️ Uploading to server...")
        cancelNetworkTask = false

        runOnUiThread {
            previewScrollView.visibility = View.GONE
            textureView.visibility       = View.VISIBLE
        }

        Thread {
            try {
                val dir       = File(sessionFolder)
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
                        "metadata", f.name,
                        f.asRequestBody("application/json".toMediaType())
                    )
                }

                val uploadClient = OkHttpClient.Builder()
                    .writeTimeout(180, java.util.concurrent.TimeUnit.SECONDS)
                    .readTimeout(30,   java.util.concurrent.TimeUnit.SECONDS)
                    .build()
                val request = Request.Builder().url(uploadUrl).post(multipart.build()).build()
                currentOkHttpCall = uploadClient.newCall(request)

                val uploadResp = try {
                    currentOkHttpCall!!.execute()
                } catch (e: Exception) {
                    if (cancelNetworkTask) return@Thread
                    updateStatus("❌ Upload failed: ${e.message}")
                    runOnUiThread { resetUploadButton() }
                    return@Thread
                }

                val uploadCode = uploadResp.code
                val uploadBody = uploadResp.body?.string() ?: ""
                uploadResp.close()

                if (!uploadResp.isSuccessful) {
                    updateStatus("❌ Upload rejected (HTTP $uploadCode): $uploadBody")
                    runOnUiThread { resetUploadButton() }
                    return@Thread
                }
                updateStatus("✅ Files received ⏳ Server processing...")

                runOnUiThread {
                    uploadButton.text = "CANCEL"
                    uploadButton.setBackgroundColor(android.graphics.Color.parseColor("#F44336"))
                    uploadButton.isEnabled = true
                    uploadButton.setOnClickListener {
                        cancelNetworkTask = true
                        currentOkHttpCall?.cancel()
                        updateStatus("⛔ Cancelled.")
                        resetUploadButton()
                    }
                }

                val pollClient = OkHttpClient.Builder()
                    .connectTimeout(10, java.util.concurrent.TimeUnit.SECONDS)
                    .readTimeout(10,   java.util.concurrent.TimeUnit.SECONDS)
                    .build()
                var elapsed = 0

                while (true) {
                    if (cancelNetworkTask) return@Thread
                    Thread.sleep(5000)
                    elapsed += 5
                    if (cancelNetworkTask) return@Thread

                    val statusResp = try {
                        pollClient.newCall(
                            Request.Builder().url(statusUrl).get().build()
                        ).execute()
                    } catch (e: Exception) {
                        updateStatus("⏳ Processing... ${elapsed}s (server unreachable, retrying)")
                        continue
                    }
                    val statusJson = try {
                        JSONObject(statusResp.body?.string() ?: "{}")
                    } catch (_: Exception) { statusResp.close(); continue }
                    statusResp.close()

                    when (statusJson.optString("status", "")) {
                        "processing" -> updateStatus(
                            "⏳ Server: ${statusJson.optString("stage")}... (${elapsed}s)")
                        "done" -> {
                            runOnUiThread {
                                uploadButton.text = "FETCH RESULTS"
                                uploadButton.setBackgroundColor(
                                    android.graphics.Color.parseColor("#2196F3"))
                                uploadButton.isEnabled = true
                                uploadButton.setOnClickListener {
                                    uploadButton.isEnabled = false
                                    downloadResult()
                                }
                            }
                            downloadResult()
                            return@Thread
                        }
                        "error" -> {
                            updateStatus("❌ Pipeline error: " +
                                statusJson.optString("error", ""))
                            runOnUiThread { resetUploadButton() }
                            return@Thread
                        }
                        else -> updateStatus("⏳ Starting... (${elapsed}s)")
                    }

                    if (elapsed > 3600) {
                        updateStatus("❌ Timeout: pipeline >1 hour. Tap SEND FILES to retry.")
                        runOnUiThread { resetUploadButton() }
                        return@Thread
                    }
                }
            } catch (e: Exception) {
                if (cancelNetworkTask) return@Thread
                updateStatus("❌ System error: ${e.message}")
                runOnUiThread { resetUploadButton() }
            }
        }.start()
    }

    private fun resetUploadButton() {
        uploadButton.text = "SEND FILES TO SERVER"
        uploadButton.setBackgroundColor(android.graphics.Color.parseColor("#4CAF50"))
        uploadButton.isEnabled = true
        uploadButton.setOnClickListener {
            uploadButton.isEnabled = false
            uploadToServer()
        }
    }

    private fun downloadResult() {
        val ip = serverIp
        if (ip == null) {
            updateStatus("❌ Server IP not known. Please reconnect.")
            runOnUiThread { uploadButton.isEnabled = true }
            return
        }
        val resultUrl = "http://$ip:$SERVER_PORT/result"

        Thread {
            if (cancelNetworkTask) return@Thread
            try {
                updateStatus("📥 Downloading result from server...")
                val client = OkHttpClient.Builder()
                    .connectTimeout(15, java.util.concurrent.TimeUnit.SECONDS)
                    .readTimeout(120, java.util.concurrent.TimeUnit.SECONDS)
                    .build()
                val resp = client.newCall(
                    Request.Builder().url(resultUrl).get().build()
                ).execute()

                if (!resp.isSuccessful) {
                    val body = resp.body?.string() ?: ""
                    resp.close()
                    updateStatus("❌ Download failed (HTTP ${resp.code}): $body")
                    runOnUiThread { resetUploadButton() }
                    return@Thread
                }
                val htmlBytes = resp.body?.bytes()
                resp.close()

                if (htmlBytes == null || htmlBytes.isEmpty()) {
                    updateStatus("❌ Empty result from server")
                    runOnUiThread { resetUploadButton() }
                    return@Thread
                }

                val zipFile = File(sessionFolder, "results.zip")
                zipFile.writeBytes(htmlBytes)
                val sizeKb = htmlBytes.size / 1024
                updateStatus("✅ Downloaded ${sizeKb} KB. Extracting...")
                extractZipToSessionFolder()
            } catch (e: Exception) {
                updateStatus("❌ Download error: ${e.message}")
                runOnUiThread { uploadButton.isEnabled = true }
            }
        }.start()
    }

    private fun extractZipToSessionFolder() {
        Thread {
            try {
                updateStatus("📦 Saving results to scan folder...")
                val zipFile = File(sessionFolder, "results.zip")

                ZipInputStream(FileInputStream(zipFile)).use { zis ->
                    var entry = zis.nextEntry
                    while (entry != null) {
                        if (!entry.isDirectory) {
                            val outFile = File(sessionFolder, entry.name)
                            FileOutputStream(outFile).use { fos ->
                                zis.copyTo(fos)
                            }
                        }
                        zis.closeEntry()
                        entry = zis.nextEntry
                    }
                }
                zipFile.delete()
                updateStatus("✅ All files saved to:\n${sessionFolder.substringAfterLast("/")}")
                runOnUiThread { uploadButton.isEnabled = true }
            } catch (e: Exception) {
                updateStatus("❌ Save error: ${e.message}")
                runOnUiThread { uploadButton.isEnabled = true }
            }
        }.start()
    }

    // ═══════════════════════════════════════════════════════════════════
    // Auto-Exposure: Setup, Probe Sequence, Computation
    // ═══════════════════════════════════════════════════════════════════

    private fun setupAutoExposureSwitch() {
        autoExposureSwitch.setOnCheckedChangeListener { _, isChecked ->
            autoExposureEnabled = isChecked
            computedShutterNs = null
        }
    }

    private fun runProbeSequence() {
        savedSessionFolderForProbe  = sessionFolder
        savedFramesPerLightForProbe = framesPerLight
        savedFormatModeForProbe     = formatMode

        probeFolderPath = "$baseFolder/probe_temp"
        File(probeFolderPath).apply { deleteRecursively(); mkdirs() }

        sessionFolder  = probeFolderPath
        framesPerLight = 1
        formatMode     = "DNG"
        isProbeMode    = true
        currentLight   = 1
        captureCount   = 0

        runOnUiThread {
            startSequenceButton.isEnabled = false
            cbLight1.isChecked = false
            cbLight2.isChecked = false
            cbLight3.isChecked = false
            cbLight4.isChecked = false
            updateStatus("🔍 Running probe capture (1 frame / light)…")
        }
        savedProbeShutterNs = shutterNsValues[currentShutterIdx]
        sendPreviewLightOff()

        Handler(Looper.getMainLooper()).postDelayed({
            sendToServer("START")
        }, 2500)
    }

    private fun onProbeSequenceComplete() {
        updateStatus("🔬 Probe captured. Computing optimal exposure…")

        Thread {
            val dir = File(probeFolderPath)
            val probePaths = dir.listFiles()
                ?.filter { it.name.endsWith(".dng", ignoreCase = true) }
                ?.sorted()
                ?.map { it.absolutePath }
                ?.toTypedArray()

            if (probePaths.isNullOrEmpty()) {
                updateStatus("⚠️ Probe failed: no DNG files found.\n" +
                             "Proceeding with manual exposure.")
                restoreAfterProbe()
                runOnUiThread {
                    startSequenceButton.text      = "START CAPTURE"
                    startSequenceButton.isEnabled = true
                }
                return@Thread
            }

            val probeMax = computeProbeMax(probePaths, probePaths.size, 99.990f)

            if (probeMax <= 0f) {
                updateStatus("⚠️ Probe returned invalid max ($probeMax).\n" +
                             "Proceeding with manual exposure.")
                restoreAfterProbe()
                runOnUiThread {
                    startSequenceButton.text      = "START CAPTURE"
                    startSequenceButton.isEnabled = true
                }
                return@Thread
            }

            if (probeMax >= sensorWhiteLevel * 0.99f) {
                restoreAfterProbe()
                updateStatus(
                    "⚠️ Probe overexposed!\n" +
                    "Move the shutter slider toward a faster speed (e.g. 1/500s)\n" +
                    "and tap RUN PROBE CAPTURE again.")
                runOnUiThread {
                    startSequenceButton.text      = "RUN PROBE CAPTURE"
                    startSequenceButton.isEnabled = true
                }
                File(probeFolderPath).deleteRecursively()
                return@Thread
            }

            val probeShutterNs = savedProbeShutterNs
            val targetValue     = 0.85f * sensorWhiteLevel
            val scale           = targetValue / probeMax
            val rawNewShutterNs = (probeShutterNs.toFloat() * scale).toLong()

            computedShutterNs = rawNewShutterNs.coerceIn(
                sensorExposureTimeMin, sensorExposureTimeMax)

            val wasClipped = rawNewShutterNs != computedShutterNs
            val newMs      = computedShutterNs!! / 1_000_000.0
            val clipNote   = if (wasClipped) " ⚠️ clamped to HW limit" else ""

            android.util.Log.i("PS_PROBE",
                "probeMax=%.0f  whiteLevel=%d  scale=%.3f  probe=%dns  computed=%.2fms%s"
                    .format(probeMax, sensorWhiteLevel, scale, probeShutterNs, newMs,
                            if (wasClipped) " CLAMPED" else ""))

            File(probeFolderPath).deleteRecursively()
            restoreAfterProbe()

            updateStatus(
                "✅ Auto-exposure done.\n" +
                "Computed shutter: %.2fms%s\n".format(newMs, clipNote) +
                "Tap START CAPTURE to proceed.")

            runOnUiThread {
                startSequenceButton.text      = "START CAPTURE (AUTO-EXPOSED)"
                startSequenceButton.isEnabled = true
                updateInfoText()
            }
        }.start()
    }

    private fun restoreAfterProbe() {
        sessionFolder  = savedSessionFolderForProbe
        framesPerLight = savedFramesPerLightForProbe
        formatMode     = savedFormatModeForProbe
        isProbeMode    = false
    }

    // ═══════════════════════════════════════════════════════════════════
    // Light Calibration
    // Uploads white-paper PNGs to /upload_calibration on the server.
    // Completely separate from the scan pipeline.
    // No masking, no reconstruction, no pipeline job started.
    // ═══════════════════════════════════════════════════════════════════

    private fun runCalibration() {
        if (finalPreviewImages.size < 4) {
            Toast.makeText(
                this,
                "Capture and process white paper images first (need 4 PNGs).",
                Toast.LENGTH_LONG
            ).show()
            return
        }

        AlertDialog.Builder(this)
            .setTitle("Calibrate Lights")
            .setMessage(
                "This will upload the current 4 PNG images to the server as a\n" +
                "white-paper calibration reference.\n\n" +
                "Make sure you captured FLAT WHITE PAPER (not an object scan)\n" +
                "before proceeding.\n\n" +
                "The server will compute new intensity correction factors and\n" +
                "update its pipeline automatically."
            )
            .setPositiveButton("CALIBRATE") { _, _ ->
                calibrateButton.isEnabled = false
                updateStatus("⬆️ Uploading calibration images...")
                Thread { uploadCalibrationImages() }.start()
            }
            .setNegativeButton("CANCEL", null)
            .show()
    }

    private fun uploadCalibrationImages() {
        val ip = serverIp
        if (ip == null) {
            updateStatus("❌ Server not connected. Cannot calibrate.")
            runOnUiThread { calibrateButton.isEnabled = true }
            return
        }

        try {
            val calibUrl  = "http://$ip:$SERVER_PORT/upload_calibration"

            val multipart = MultipartBody.Builder().setType(MultipartBody.FORM)

            for (path in finalPreviewImages) {
                val f = File(path)
                if (!f.exists()) {
                    updateStatus("❌ PNG not found: ${f.name}")
                    runOnUiThread { calibrateButton.isEnabled = true }
                    return
                }
                multipart.addFormDataPart(
                    "file", f.name,
                    f.asRequestBody("image/png".toMediaType())
                )
            }

            val client = OkHttpClient.Builder()
                .connectTimeout(15, java.util.concurrent.TimeUnit.SECONDS)
                .writeTimeout(120, java.util.concurrent.TimeUnit.SECONDS)
                .readTimeout(30,   java.util.concurrent.TimeUnit.SECONDS)
                .build()

            val request = Request.Builder()
                .url(calibUrl)
                .post(multipart.build())
                .build()

            updateStatus("⏳ Server computing calibration factors...")

            val response = client.newCall(request).execute()
            val responseCode = response.code
            val responseBody = response.body?.string() ?: ""
            response.close()

            if (responseCode != 200) {
                updateStatus("❌ Calibration failed (HTTP $responseCode): $responseBody")
                runOnUiThread { calibrateButton.isEnabled = true }
                return
            }

            val resultJson = try {
                JSONObject(responseBody)
            } catch (e: Exception) {
                updateStatus("❌ Invalid response from server: $responseBody")
                runOnUiThread { calibrateButton.isEnabled = true }
                return
            }

            runOnUiThread {
                calibrateButton.isEnabled = true
                showCalibrationResult(resultJson)
            }

        } catch (e: Exception) {
            updateStatus("❌ Calibration upload error: ${e.message}")
            runOnUiThread { calibrateButton.isEnabled = true }
        }
    }

    private fun showCalibrationResult(result: JSONObject) {
        val status = result.optString("status", "error")

        if (status != "ok") {
            AlertDialog.Builder(this)
                .setTitle("Calibration Failed")
                .setMessage(
                    "The server reported an error:\n\n" +
                    result.optString("message", "Unknown error")
                )
                .setPositiveButton("OK", null)
                .show()
            updateStatus("❌ Calibration failed — see dialog.")
            return
        }

        val factorsArray = result.optJSONArray("factors")
        val factorsText = if (factorsArray != null && factorsArray.length() == 4) {
            buildString {
                appendLine("  Light 1:  %.4f".format(factorsArray.getDouble(0)))
                appendLine("  Light 2:  %.4f".format(factorsArray.getDouble(1)))
                appendLine("  Light 3:  %.4f".format(factorsArray.getDouble(2)))
                append(    "  Light 4:  %.4f".format(factorsArray.getDouble(3)))
            }
        } else {
            "(factors not returned in response)"
        }

        AlertDialog.Builder(this)
            .setTitle("✅ Calibration Complete")
            .setMessage(
                "New intensity correction factors:\n\n" +
                "$factorsText\n\n" +
                "process_pipeline.py on the server has been updated automatically.\n\n" +
                "The next scan will use the new calibration."
            )
            .setPositiveButton("OK", null)
            .show()

        updateStatus("✅ Calibration complete — factors updated on server.")
    }

    // ═══════════════════════════════════════════════════════════════════
    // Companion Object
    // ═══════════════════════════════════════════════════════════════════

    companion object {
        init { System.loadLibrary("ps_engine") }
        private const val CAMERA_ID           = "0"
        private const val SENSOR_WIDTH        = 4032
        private const val SENSOR_HEIGHT       = 3024
        private const val HIST_SAMPLE_EVERY   = 10
        private const val CAMERA_REQUEST_CODE = 100
        private const val SERVER_PORT         = 8080
        private const val CAPTURE_DELAY_MS    = 1000L
    }

    external fun processAndDenoise(
        filePaths: Array<String>, numFiles: Int,
        outputPath: String, rotationDegrees: Int,
        scalingFactor: Float
    ): Int

    external fun computeProbeMax(
        filePaths: Array<String>, numFiles: Int, percentile: Float
    ): Float
}
