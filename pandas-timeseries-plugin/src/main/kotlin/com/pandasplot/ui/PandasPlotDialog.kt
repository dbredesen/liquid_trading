package com.pandasplot.ui

import com.intellij.openapi.progress.ProgressIndicator
import com.intellij.openapi.progress.ProgressManager
import com.intellij.openapi.progress.Task
import com.intellij.openapi.project.Project
import com.intellij.openapi.ui.DialogWrapper
import com.intellij.ui.CollectionListModel
import com.intellij.ui.ColoredListCellRenderer
import com.intellij.ui.SimpleTextAttributes
import com.intellij.ui.components.JBList
import com.intellij.ui.components.JBScrollPane
import com.pandasplot.debugger.DataFrameEvaluator
import com.pandasplot.debugger.DataFrameInfo
import com.pandasplot.debugger.X_AXIS_INDEX
import com.pandasplot.debugger.X_AXIS_ROWNUM
import java.awt.BorderLayout
import java.awt.Dimension
import java.awt.FlowLayout
import java.awt.GridBagConstraints
import java.awt.GridBagLayout
import java.awt.Insets
import javax.swing.BorderFactory
import javax.swing.DefaultComboBoxModel
import javax.swing.JButton
import javax.swing.JComboBox
import javax.swing.JComponent
import javax.swing.JLabel
import javax.swing.JList
import javax.swing.JPanel
import javax.swing.JSplitPane
import javax.swing.ListSelectionModel
import javax.swing.SwingUtilities

/**
 * Main dialog for the Pandas Time Series Plot plugin.
 *
 * Layout:
 * ┌──────────────────────────────────────────────────────────────┐
 * │  Controls panel (top)                                        │
 * │  ┌─────────────────────────────────────────────────────┐    │
 * │  │ DataFrame: [combobox]                               │    │
 * │  │ X axis (timestamp): [combobox]                      │    │
 * │  │ Y axis (series):    [list – multi-select]  [Plot ▶] │    │
 * │  └─────────────────────────────────────────────────────┘    │
 * ├──────────────────────────────────────────────────────────────┤
 * │  Chart panel (bottom, resizable)                             │
 * └──────────────────────────────────────────────────────────────┘
 */
class PandasPlotDialog(
    private val project: Project,
    private val dataFrames: List<DataFrameInfo>
) : DialogWrapper(project, true) {

    // ------- UI components -------
    private val dfCombo = JComboBox(DefaultComboBoxModel(dataFrames.map { it.name }.toTypedArray()))
    private val xAxisCombo = JComboBox<String>()
    private val yAxisList = JBList<String>()
    private val chartPanel = TimeSeriesChartPanel()
    private val statusLabel = JLabel(" ")

    // Currently displayed DataFrame
    private var currentDf: DataFrameInfo? = null

    init {
        title = "Pandas Time Series Plot"
        setOKButtonText("Close")
        setCancelButtonText("Reset")
        isModal = false      // non-modal so the user can interact with PyCharm
        init()
        // Populate controls for the first DataFrame
        if (dataFrames.isNotEmpty()) {
            onDataFrameSelected(dataFrames[0])
        }
    }

    override fun createCenterPanel(): JComponent {
        val controlsPanel = buildControlsPanel()
        val chartWrapper = JPanel(BorderLayout()).apply {
            add(chartPanel, BorderLayout.CENTER)
            add(statusLabel.also { it.border = BorderFactory.createEmptyBorder(2, 6, 2, 6) },
                BorderLayout.SOUTH)
        }

        return JSplitPane(JSplitPane.VERTICAL_SPLIT, controlsPanel, chartWrapper).apply {
            resizeWeight = 0.0          // controls keep fixed size; chart gets extra space
            dividerSize = 5
            isContinuousLayout = true
            preferredSize = Dimension(900, 650)
        }
    }

    // ------- Controls panel -------

    private fun buildControlsPanel(): JPanel {
        val panel = JPanel(GridBagLayout()).apply {
            border = BorderFactory.createCompoundBorder(
                BorderFactory.createTitledBorder("Data Selection"),
                BorderFactory.createEmptyBorder(4, 8, 4, 8)
            )
        }

        val gc = GridBagConstraints().apply {
            insets = Insets(4, 4, 4, 4)
            fill = GridBagConstraints.HORIZONTAL
            anchor = GridBagConstraints.WEST
        }

        // Row 0: DataFrame selector
        gc.gridx = 0; gc.gridy = 0; gc.weightx = 0.0
        panel.add(JLabel("DataFrame:"), gc)
        gc.gridx = 1; gc.weightx = 1.0
        panel.add(dfCombo.also { combo ->
            combo.addActionListener {
                val name = combo.selectedItem as? String ?: return@addActionListener
                val df = dataFrames.find { it.name == name } ?: return@addActionListener
                onDataFrameSelected(df)
            }
        }, gc)

        // Row 0 info label (row/col count)
        val dfInfoLabel = JLabel("")
        gc.gridx = 2; gc.weightx = 0.0
        panel.add(dfInfoLabel, gc)

        // Row 1: X-axis column
        gc.gridx = 0; gc.gridy = 1; gc.weightx = 0.0
        panel.add(JLabel("X axis:"), gc)
        gc.gridx = 1; gc.weightx = 1.0
        panel.add(xAxisCombo, gc)

        // Row 2: Y-axis columns (multi-select list)
        gc.gridx = 0; gc.gridy = 2; gc.weightx = 0.0; gc.anchor = GridBagConstraints.NORTHWEST
        panel.add(JLabel("Y axis (series):"), gc)

        yAxisList.apply {
            selectionMode = ListSelectionModel.MULTIPLE_INTERVAL_SELECTION
            visibleRowCount = 6
            cellRenderer = ColumnListCellRenderer()
        }

        gc.gridx = 1; gc.weightx = 1.0; gc.fill = GridBagConstraints.BOTH
        panel.add(JBScrollPane(yAxisList).apply {
            preferredSize = Dimension(300, 130)
        }, gc)

        // Row 2: Plot button (beside Y list)
        val plotButton = JButton("Plot ▶").apply {
            addActionListener { onPlotClicked() }
        }
        gc.gridx = 2; gc.weightx = 0.0; gc.fill = GridBagConstraints.NONE
        gc.anchor = GridBagConstraints.NORTH
        panel.add(plotButton, gc)

        // Update info label when selection changes
        dfCombo.addActionListener {
            val df = currentDf ?: return@addActionListener
            dfInfoLabel.text = "${df.rowCount} rows × ${df.shape.second} cols"
        }

        return panel
    }

    // ------- Event handlers -------

    private fun onDataFrameSelected(df: DataFrameInfo) {
        currentDf = df

        // Populate X axis combo:
        //   1. <rownum>  – always available
        //   2. "DataFrame index" – when the index has a datetime dtype
        //   3. Any column whose dtype is a datetime/timestamp type
        xAxisCombo.removeAllItems()
        xAxisCombo.addItem(X_AXIS_ROWNUM)
        if (df.isIndexDateTime) {
            xAxisCombo.addItem(X_AXIS_INDEX)
        }
        df.timestampColumns.forEach { xAxisCombo.addItem(it) }

        // Populate Y axis list (default to numeric columns)
        val model = CollectionListModel(df.columns)
        yAxisList.model = model
        // Pre-select all numeric columns
        val numericIndices = df.numericColumns
            .mapNotNull { col -> df.columns.indexOf(col).takeIf { it >= 0 } }
            .toIntArray()
        if (numericIndices.isNotEmpty()) {
            yAxisList.selectedIndices = numericIndices
        }

        statusLabel.text = "DataFrame '${df.name}': ${df.rowCount} rows × ${df.shape.second} columns"
    }

    private fun onPlotClicked() {
        val df = currentDf ?: return
        val xCol = xAxisCombo.selectedItem as? String ?: return
        val yCols = yAxisList.selectedValuesList
        if (yCols.isEmpty()) {
            statusLabel.text = "Please select at least one Y-axis column."
            return
        }

        statusLabel.text = "Loading data…"

        ProgressManager.getInstance().run(
            object : Task.Backgroundable(project, "Extracting DataFrame data…", false) {

                override fun run(indicator: ProgressIndicator) {
                    indicator.isIndeterminate = true
                    val evaluator = DataFrameEvaluator(project)
                    val data = evaluator.extractTimeSeriesData(df.name, xCol, yCols)

                    SwingUtilities.invokeLater {
                        if (data == null) {
                            statusLabel.text = "Failed to extract data from '${df.name}'. " +
                                "Check that the DataFrame is still in scope."
                            chartPanel.clear()
                        } else {
                            chartPanel.plot(data)
                            statusLabel.text =
                                "Plotted ${data.series.size} series, ${data.timestamps.size} points" +
                                " — X: '${data.timestampColumn}'"
                        }
                    }
                }
            }
        )
    }

    override fun doCancelAction() {
        // "Reset" button clears the chart
        chartPanel.clear()
        statusLabel.text = " "
    }

    // ------- Custom list renderer -------

    /**
     * Renders column names with their dtype appended in a lighter style.
     * Timestamp columns get a calendar prefix; numeric columns get a tilde prefix.
     */
    private inner class ColumnListCellRenderer : ColoredListCellRenderer<String>() {
        override fun customizeCellRenderer(
            list: JList<out String>,
            value: String,
            index: Int,
            selected: Boolean,
            hasFocus: Boolean
        ) {
            val df = currentDf ?: return
            val dtype = df.dtypes[value] ?: "?"
            val isTimestamp = value in df.timestampColumns
            val isNumeric = value in df.numericColumns

            val prefix = when {
                isTimestamp -> "📅 "
                isNumeric   -> "~ "
                else        -> "  "
            }

            append(prefix, SimpleTextAttributes.GRAYED_ATTRIBUTES)
            append(value, SimpleTextAttributes.REGULAR_ATTRIBUTES)
            append("  $dtype", SimpleTextAttributes.GRAYED_SMALL_ATTRIBUTES)
        }
    }
}
