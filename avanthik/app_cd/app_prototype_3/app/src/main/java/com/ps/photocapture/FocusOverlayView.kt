package com.ps.photocapture

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View

/**
 * Transparent overlay drawn on top of the TextureView.
 * Handles:
 *   1. Focus rectangle (yellow while focusing, green when locked, fades out)
 *   2. RGB histogram (drawn in bottom-right corner)
 */
class FocusOverlayView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null
) : View(context, attrs) {

    // ── Focus rectangle state ─────────────────────────────────────────
    enum class FocusState { IDLE, FOCUSING, LOCKED, FAILED }

    private var focusState      = FocusState.IDLE
    private var focusRect       = RectF()
    private var focusAlpha      = 255
    private var fadeRunning     = false

    // ── Histogram data ────────────────────────────────────────────────
    // Each array has 256 entries — one per brightness level
    private var histR    = IntArray(256)
    private var histG    = IntArray(256)
    private var histB    = IntArray(256)
    private var hasHist  = false

    // ── Paints ────────────────────────────────────────────────────────
    private val focusPaintYellow = Paint().apply {
        style       = Paint.Style.STROKE
        color       = Color.YELLOW
        strokeWidth = 3f
        isAntiAlias = true
    }
    private val focusPaintGreen = Paint().apply {
        style       = Paint.Style.STROKE
        color       = Color.GREEN
        strokeWidth = 3f
        isAntiAlias = true
    }
    private val cornerPaint = Paint().apply {
        style       = Paint.Style.STROKE
        strokeWidth = 5f
        isAntiAlias = true
        strokeCap   = Paint.Cap.ROUND
    }
    private val histBgPaint = Paint().apply {
        color = Color.argb(160, 0, 0, 0)
        style = Paint.Style.FILL
    }
    private val histPaintR = Paint().apply {
        color       = Color.argb(180, 255, 60, 60)
        style       = Paint.Style.FILL
        isAntiAlias = true
    }
    private val histPaintG = Paint().apply {
        color       = Color.argb(180, 60, 255, 60)
        style       = Paint.Style.FILL
        isAntiAlias = true
    }
    private val histPaintB = Paint().apply {
        color       = Color.argb(180, 60, 120, 255)
        style       = Paint.Style.FILL
        isAntiAlias = true
    }
    private val histBorderPaint = Paint().apply {
        color       = Color.argb(200, 255, 255, 255)
        style       = Paint.Style.STROKE
        strokeWidth = 1.5f
    }

    // ── Focus API ─────────────────────────────────────────────────────

    /** Call when AF trigger fires — shows yellow box at centre */
    fun showFocusing() {
        focusState  = FocusState.FOCUSING
        focusAlpha  = 255
        fadeRunning = false
        // Default rect = centre 20% of view
        val cx = width  / 2f
        val cy = height / 2f
        val sz = width  * 0.15f
        focusRect = RectF(cx - sz, cy - sz, cx + sz, cy + sz)
        invalidate()
    }

    /** Call when AF locks successfully */
    fun showFocusLocked() {
        focusState  = FocusState.LOCKED
        focusAlpha  = 255
        invalidate()
        // Fade out after 1.5s
        postDelayed({ startFade() }, 1500)
    }

    /** Call when AF fails */
    fun showFocusFailed() {
        focusState = FocusState.FAILED
        focusAlpha = 255
        invalidate()
        postDelayed({ resetFocus() }, 1000)
    }

    fun resetFocus() {
        focusState = FocusState.IDLE
        invalidate()
    }

    private fun startFade() {
        if (fadeRunning) return
        fadeRunning = true
        fadeTick()
    }

    private fun fadeTick() {
        focusAlpha -= 25
        if (focusAlpha <= 0) {
            focusAlpha  = 0
            focusState  = FocusState.IDLE
            fadeRunning = false
        }
        invalidate()
        if (focusState != FocusState.IDLE) {
            postDelayed({ fadeTick() }, 50)
        }
    }

    // ── Histogram API ─────────────────────────────────────────────────

    /** Called from background thread — post to UI thread via invalidate */
    fun updateHistogram(r: IntArray, g: IntArray, b: IntArray) {
        histR   = r.copyOf()
        histG   = g.copyOf()
        histB   = b.copyOf()
        hasHist = true
        postInvalidate() // safe to call from background thread
    }

    // ── Draw ──────────────────────────────────────────────────────────

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        drawFocusRect(canvas)
        if (hasHist) drawHistogram(canvas)
    }

    private fun drawFocusRect(canvas: Canvas) {
        if (focusState == FocusState.IDLE) return
        if (focusRect.isEmpty) return

        val paint = when (focusState) {
            FocusState.FOCUSING -> focusPaintYellow.also { it.alpha = focusAlpha }
            FocusState.LOCKED   -> focusPaintGreen.also  { it.alpha = focusAlpha }
            FocusState.FAILED   -> focusPaintYellow.also { it.alpha = focusAlpha; it.color = Color.RED }
            else                -> return
        }

        // Draw the main rectangle
        canvas.drawRect(focusRect, paint)

        // Draw corner decorations (L-shaped brackets)
        val cLen = focusRect.width() * 0.2f
        cornerPaint.color = paint.color
        cornerPaint.alpha = focusAlpha

        // Top-left
        canvas.drawLine(focusRect.left, focusRect.top, focusRect.left + cLen, focusRect.top, cornerPaint)
        canvas.drawLine(focusRect.left, focusRect.top, focusRect.left, focusRect.top + cLen, cornerPaint)
        // Top-right
        canvas.drawLine(focusRect.right, focusRect.top, focusRect.right - cLen, focusRect.top, cornerPaint)
        canvas.drawLine(focusRect.right, focusRect.top, focusRect.right, focusRect.top + cLen, cornerPaint)
        // Bottom-left
        canvas.drawLine(focusRect.left, focusRect.bottom, focusRect.left + cLen, focusRect.bottom, cornerPaint)
        canvas.drawLine(focusRect.left, focusRect.bottom, focusRect.left, focusRect.bottom - cLen, cornerPaint)
        // Bottom-right
        canvas.drawLine(focusRect.right, focusRect.bottom, focusRect.right - cLen, focusRect.bottom, cornerPaint)
        canvas.drawLine(focusRect.right, focusRect.bottom, focusRect.right, focusRect.bottom - cLen, cornerPaint)
    }

    private fun drawHistogram(canvas: Canvas) {
        val histW    = width  * 0.42f
        val histH    = height * 0.14f
        val margin   = 12f
        val left     = width  - histW - margin
        val top      = height - histH - margin
        val right    = width  - margin
        val bottom   = height - margin

        // Background
        canvas.drawRoundRect(left - 4, top - 4, right + 4, bottom + 4, 8f, 8f, histBgPaint)
        canvas.drawRoundRect(left - 4, top - 4, right + 4, bottom + 4, 8f, 8f, histBorderPaint)

        val maxVal = maxOf(histR.max(), histG.max(), histB.max()).coerceAtLeast(1)
        val binW   = (right - left) / 256f

        // Draw R, G, B bars
        for (i in 0..255) {
            val x1 = left + i * binW
            val x2 = x1 + binW

            val rH = (histR[i].toFloat() / maxVal) * histH
            val gH = (histG[i].toFloat() / maxVal) * histH
            val bH = (histB[i].toFloat() / maxVal) * histH

            if (rH > 0) canvas.drawRect(x1, bottom - rH, x2, bottom, histPaintR)
            if (gH > 0) canvas.drawRect(x1, bottom - gH, x2, bottom, histPaintG)
            if (bH > 0) canvas.drawRect(x1, bottom - bH, x2, bottom, histPaintB)
        }
    }
}