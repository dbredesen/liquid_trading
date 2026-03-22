package com.pandasplot.ui

import com.pandasplot.debugger.TimeSeriesData
import org.jfree.chart.ChartFactory
import org.jfree.chart.ChartPanel
import org.jfree.chart.JFreeChart
import org.jfree.chart.axis.DateAxis
import org.jfree.chart.axis.NumberAxis
import org.jfree.chart.plot.XYPlot
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer
import org.jfree.chart.ui.RectangleInsets
import org.jfree.data.time.Millisecond
import org.jfree.data.time.TimeSeries
import org.jfree.data.time.TimeSeriesCollection
import java.awt.BasicStroke
import java.awt.Color
import java.awt.Dimension
import java.text.ParseException
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import javax.swing.BorderFactory
import javax.swing.JPanel
import java.awt.BorderLayout

/** Distinct colours for the plotted series (cycles if more than 8 series). */
private val SERIES_COLORS = listOf(
    Color(0x4A90D9),  // blue
    Color(0xE8832A),  // orange
    Color(0x2CA02C),  // green
    Color(0xD62728),  // red
    Color(0x9467BD),  // purple
    Color(0x8C564B),  // brown
    Color(0xE377C2),  // pink
    Color(0x7F7F7F),  // grey
)

/** Candidate datetime formats – tried in order until one succeeds. */
private val TIMESTAMP_FORMATS = listOf(
    "yyyy-MM-dd'T'HH:mm:ss.SSSSSS",
    "yyyy-MM-dd'T'HH:mm:ss.SSS",
    "yyyy-MM-dd'T'HH:mm:ss",
    "yyyy-MM-dd HH:mm:ss.SSSSSS",
    "yyyy-MM-dd HH:mm:ss.SSS",
    "yyyy-MM-dd HH:mm:ss",
    "yyyy-MM-dd",
    "MM/dd/yyyy HH:mm:ss",
    "MM/dd/yyyy",
    "dd-MM-yyyy",
)

/**
 * A Swing panel that renders a JFreeChart time-series chart from [TimeSeriesData].
 * Supports zoom (mouse-wheel), pan, and crosshair tooltips out of the box via ChartPanel.
 */
class TimeSeriesChartPanel : JPanel(BorderLayout()) {

    private var chartPanel: ChartPanel? = null

    init {
        preferredSize = Dimension(800, 400)
        showPlaceholder()
    }

    // ------------------------------------------------------------------
    // Public
    // ------------------------------------------------------------------

    /** Render [data] as an interactive time series chart. */
    fun plot(data: TimeSeriesData) {
        val dataset = buildDataset(data)
        val chart = buildChart(data.dfName, data.timestampColumn, dataset)
        replaceChart(chart)
    }

    /** Clear the chart area and show the placeholder message. */
    fun clear() {
        removeAll()
        showPlaceholder()
        revalidate()
        repaint()
    }

    // ------------------------------------------------------------------
    // Private helpers
    // ------------------------------------------------------------------

    private fun showPlaceholder() {
        val label = javax.swing.JLabel(
            "Select columns and click \"Plot\" to display the chart.",
            javax.swing.SwingConstants.CENTER
        )
        label.foreground = Color.GRAY
        add(label, BorderLayout.CENTER)
    }

    private fun replaceChart(chart: JFreeChart) {
        removeAll()
        chartPanel = ChartPanel(chart).apply {
            isMouseWheelEnabled = true        // zoom with scroll wheel
            isDomainZoomable = true
            isRangeZoomable = true
            border = BorderFactory.createEmptyBorder(4, 4, 4, 4)
        }
        add(chartPanel!!, BorderLayout.CENTER)
        revalidate()
        repaint()
    }

    private fun buildDataset(data: TimeSeriesData): TimeSeriesCollection {
        val collection = TimeSeriesCollection()

        data.series.forEach { (colName, values) ->
            val ts = TimeSeries(colName)
            data.timestamps.forEachIndexed { idx, tsStr ->
                val date = parseTimestamp(tsStr) ?: return@forEachIndexed
                val value = values.getOrNull(idx) ?: return@forEachIndexed
                try {
                    ts.addOrUpdate(Millisecond(date), value)
                } catch (_: Exception) { /* duplicate timestamp – skip */ }
            }
            collection.addSeries(ts)
        }

        return collection
    }

    private fun buildChart(title: String, xLabel: String, dataset: TimeSeriesCollection): JFreeChart {
        val chart = ChartFactory.createTimeSeriesChart(
            title,          // chart title
            xLabel,         // x-axis label
            "Value",        // y-axis label
            dataset,
            true,           // legend
            true,           // tooltips
            false           // URLs
        )

        styleChart(chart)
        return chart
    }

    private fun styleChart(chart: JFreeChart) {
        chart.backgroundPaint = Color.WHITE
        chart.padding = RectangleInsets(8.0, 8.0, 8.0, 8.0)

        val plot = chart.plot as XYPlot
        plot.backgroundPaint = Color(0xF8F8F8)
        plot.domainGridlinePaint = Color(0xDDDDDD)
        plot.rangeGridlinePaint = Color(0xDDDDDD)
        plot.outlineVisible = false
        plot.insets = RectangleInsets(4.0, 4.0, 4.0, 4.0)

        // Style axes
        (plot.domainAxis as? DateAxis)?.apply {
            tickLabelFont = tickLabelFont.deriveFont(10f)
            labelFont = labelFont.deriveFont(11f)
            isVerticalTickLabels = true
        }
        (plot.rangeAxis as? NumberAxis)?.apply {
            tickLabelFont = tickLabelFont.deriveFont(10f)
            labelFont = labelFont.deriveFont(11f)
            autoRangeIncludesZero = false
        }

        // Style each series with a distinct colour and a slightly thicker line
        val renderer = XYLineAndShapeRenderer()
        for (i in 0 until (plot.dataset?.seriesCount ?: 0)) {
            val color = SERIES_COLORS[i % SERIES_COLORS.size]
            renderer.setSeriesPaint(i, color)
            renderer.setSeriesStroke(i, BasicStroke(1.8f))
            renderer.setSeriesShapesVisible(i, false)  // hide per-point markers by default
        }
        plot.renderer = renderer

        // Enable crosshairs
        plot.isDomainCrosshairVisible = true
        plot.isRangeCrosshairVisible = true
        plot.domainCrosshairPaint = Color(0xAAAAAA)
        plot.rangeCrosshairPaint = Color(0xAAAAAA)
    }

    // ------------------------------------------------------------------
    // Timestamp parsing
    // ------------------------------------------------------------------

    private val parsers: List<SimpleDateFormat> = TIMESTAMP_FORMATS.map {
        SimpleDateFormat(it, Locale.US).also { sdf -> sdf.isLenient = false }
    }

    /** Try to parse a timestamp string produced by Pandas [.astype(str)]. */
    private fun parseTimestamp(value: String): Date? {
        val trimmed = value.trim()

        // Numeric epoch (seconds or milliseconds)
        trimmed.toLongOrNull()?.let { epoch ->
            return if (epoch > 1_000_000_000_000L) Date(epoch) else Date(epoch * 1000L)
        }
        trimmed.toDoubleOrNull()?.let { epoch ->
            return Date((epoch * 1000L).toLong())
        }

        // String format – try each parser
        for (parser in parsers) {
            try {
                return parser.parse(trimmed)
            } catch (_: ParseException) { }
        }

        return null
    }
}
