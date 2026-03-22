package com.pandasplot.ui

import com.intellij.openapi.application.ApplicationManager
import com.intellij.openapi.progress.ProgressIndicator
import com.intellij.openapi.progress.ProgressManager
import com.intellij.openapi.progress.Task
import com.intellij.openapi.project.Project
import com.intellij.openapi.ui.Messages
import com.intellij.openapi.wm.ToolWindow
import com.intellij.openapi.wm.ToolWindowFactory
import com.intellij.openapi.wm.ToolWindowManager
import com.intellij.ui.content.ContentFactory
import com.intellij.xdebugger.XDebuggerManager
import com.intellij.xdebugger.XDebuggerManagerListener
import com.intellij.xdebugger.XDebugSession
import com.pandasplot.debugger.DataFrameEvaluator
import com.pandasplot.debugger.DataFrameInfo
import com.pandasplot.debugger.X_AXIS_INDEX
import com.pandasplot.debugger.X_AXIS_ROWNUM
import java.awt.BorderLayout
import java.awt.FlowLayout
import javax.swing.JButton
import javax.swing.JLabel
import javax.swing.JPanel

/**
 * Factory that creates the "Pandas Plot" tool window content.
 *
 * The tool window provides an always-visible panel inside the IDE debugger area.
 * It subscribes to debugger pause events and refreshes automatically when the
 * debugger stops at a breakpoint.
 */
class PandasPlotToolWindowFactory : ToolWindowFactory {

    override fun createToolWindowContent(project: Project, toolWindow: ToolWindow) {
        val panel = PandasPlotToolWindowPanel(project)

        val content = ContentFactory.getInstance()
            .createContent(panel, "Time Series", false)
        toolWindow.contentManager.addContent(content)

        // Listen for debugger session events to auto-refresh
        project.messageBus.connect().subscribe(
            XDebuggerManager.TOPIC,
            object : XDebuggerManagerListener {
                override fun processStopped(debugProcess: com.intellij.xdebugger.XDebugProcess) {
                    // Triggered when the debugger pauses (breakpoint hit, step, etc.)
                    panel.onDebuggerPaused()
                }

                override fun currentSessionChanged(
                    previousSession: XDebugSession?,
                    currentSession: XDebugSession?
                ) {
                    if (currentSession == null) panel.onDebuggerStopped()
                }
            }
        )
    }

    override fun shouldBeAvailable(project: Project): Boolean = true
}

/**
 * The actual Swing panel hosted inside the tool window.
 */
private class PandasPlotToolWindowPanel(private val project: Project) : JPanel(BorderLayout()) {

    private val chartPanel = TimeSeriesChartPanel()
    private val controlsHolder = JPanel(BorderLayout())
    private val statusLabel = JLabel("Waiting for debugger to pause…")
    private val refreshButton = JButton("Refresh DataFrames").apply {
        isEnabled = false
        addActionListener { onDebuggerPaused() }
    }

    private var columnSelectorPanel: EmbeddedColumnSelectorPanel? = null

    init {
        val topBar = JPanel(FlowLayout(FlowLayout.LEFT, 6, 4)).apply {
            add(refreshButton)
            add(statusLabel)
        }
        add(topBar, BorderLayout.NORTH)
        add(controlsHolder, BorderLayout.CENTER)
        add(chartPanel.also { it.preferredSize = java.awt.Dimension(800, 300) }, BorderLayout.SOUTH)
    }

    fun onDebuggerPaused() {
        refreshButton.isEnabled = true
        statusLabel.text = "Scanning for DataFrames…"

        ProgressManager.getInstance().run(
            object : Task.Backgroundable(project, "Scanning for DataFrames…", false) {
                private var frames: List<DataFrameInfo> = emptyList()

                override fun run(indicator: ProgressIndicator) {
                    frames = DataFrameEvaluator(project).findDataFrames()
                }

                override fun onSuccess() {
                    ApplicationManager.getApplication().invokeLater {
                        if (frames.isEmpty()) {
                            statusLabel.text = "No DataFrames found in current scope."
                            controlsHolder.removeAll()
                            controlsHolder.revalidate()
                            return@invokeLater
                        }
                        statusLabel.text = "${frames.size} DataFrame(s) found."
                        showColumnSelector(frames)
                    }
                }
            }
        )
    }

    fun onDebuggerStopped() {
        refreshButton.isEnabled = false
        statusLabel.text = "Waiting for debugger to pause…"
        controlsHolder.removeAll()
        controlsHolder.revalidate()
        chartPanel.clear()
    }

    private fun showColumnSelector(dataFrames: List<DataFrameInfo>) {
        controlsHolder.removeAll()
        val selector = EmbeddedColumnSelectorPanel(project, dataFrames, chartPanel, statusLabel)
        columnSelectorPanel = selector
        controlsHolder.add(selector, BorderLayout.CENTER)
        controlsHolder.revalidate()
        controlsHolder.repaint()
    }
}

/**
 * Compact column-selection controls embedded inside the tool window panel.
 * Reuses [TimeSeriesChartPanel] for rendering.
 */
private class EmbeddedColumnSelectorPanel(
    private val project: Project,
    dataFrames: List<DataFrameInfo>,
    private val chart: TimeSeriesChartPanel,
    private val status: JLabel
) : JPanel() {

    private val dfCombo = javax.swing.JComboBox(
        javax.swing.DefaultComboBoxModel(dataFrames.map { it.name }.toTypedArray())
    )
    private val xAxisCombo = javax.swing.JComboBox<String>()
    private val yAxisList = com.intellij.ui.components.JBList<String>()
    private var currentDf: DataFrameInfo? = null

    init {
        layout = java.awt.GridBagLayout()
        border = javax.swing.BorderFactory.createEmptyBorder(4, 8, 4, 8)
        val gc = java.awt.GridBagConstraints().apply {
            insets = java.awt.Insets(3, 3, 3, 3)
            fill = java.awt.GridBagConstraints.HORIZONTAL
            anchor = java.awt.GridBagConstraints.WEST
        }

        gc.gridx = 0; gc.gridy = 0; gc.weightx = 0.0
        add(JLabel("DataFrame:"), gc)
        gc.gridx = 1; gc.weightx = 1.0
        add(dfCombo, gc)

        gc.gridx = 0; gc.gridy = 1; gc.weightx = 0.0
        add(JLabel("X axis:"), gc)
        gc.gridx = 1; gc.weightx = 1.0
        add(xAxisCombo, gc)

        gc.gridx = 0; gc.gridy = 2; gc.weightx = 0.0; gc.anchor = java.awt.GridBagConstraints.NORTHWEST
        add(JLabel("Y axis:"), gc)
        yAxisList.selectionMode = javax.swing.ListSelectionModel.MULTIPLE_INTERVAL_SELECTION
        yAxisList.visibleRowCount = 5
        gc.gridx = 1; gc.weightx = 1.0; gc.fill = java.awt.GridBagConstraints.BOTH
        add(com.intellij.ui.components.JBScrollPane(yAxisList).apply {
            preferredSize = java.awt.Dimension(260, 110)
        }, gc)

        val plotBtn = JButton("Plot ▶").apply {
            addActionListener { onPlot() }
        }
        gc.gridx = 1; gc.gridy = 3; gc.fill = java.awt.GridBagConstraints.NONE
        gc.anchor = java.awt.GridBagConstraints.EAST; gc.weightx = 0.0
        add(plotBtn, gc)

        dfCombo.addActionListener {
            val name = dfCombo.selectedItem as? String ?: return@addActionListener
            val df = dataFrames.find { it.name == name } ?: return@addActionListener
            populate(df)
        }

        if (dataFrames.isNotEmpty()) populate(dataFrames[0])
    }

    private fun populate(df: DataFrameInfo) {
        currentDf = df
        xAxisCombo.removeAllItems()
        xAxisCombo.addItem(X_AXIS_ROWNUM)
        if (df.isIndexDateTime) {
            xAxisCombo.addItem(X_AXIS_INDEX)
        }
        df.timestampColumns.forEach { xAxisCombo.addItem(it) }

        val model = com.intellij.ui.CollectionListModel(df.columns)
        yAxisList.model = model
        val indices = df.numericColumns.mapNotNull { df.columns.indexOf(it).takeIf { i -> i >= 0 } }.toIntArray()
        if (indices.isNotEmpty()) yAxisList.selectedIndices = indices
    }

    private fun onPlot() {
        val df = currentDf ?: return
        val xCol = xAxisCombo.selectedItem as? String ?: return
        val yCols = yAxisList.selectedValuesList
        if (yCols.isEmpty()) { status.text = "Select at least one Y-axis column."; return }
        status.text = "Loading…"

        ProgressManager.getInstance().run(
            object : Task.Backgroundable(project, "Extracting data…", false) {
                override fun run(indicator: ProgressIndicator) {
                    val data = DataFrameEvaluator(project).extractTimeSeriesData(df.name, xCol, yCols)
                    javax.swing.SwingUtilities.invokeLater {
                        if (data == null) {
                            status.text = "Failed to extract data."
                            chart.clear()
                        } else {
                            chart.plot(data)
                            status.text = "Plotted ${data.series.size} series (${data.timestamps.size} pts)"
                        }
                    }
                }
            }
        )
    }
}
