package com.pandasplot.actions

import com.intellij.openapi.actionSystem.AnAction
import com.intellij.openapi.actionSystem.AnActionEvent
import com.intellij.openapi.application.ApplicationManager
import com.intellij.openapi.progress.ProgressIndicator
import com.intellij.openapi.progress.ProgressManager
import com.intellij.openapi.progress.Task
import com.intellij.openapi.ui.Messages
import com.intellij.xdebugger.XDebuggerManager
import com.pandasplot.debugger.DataFrameEvaluator
import com.pandasplot.debugger.DataFrameInfo
import com.pandasplot.ui.PandasPlotDialog

/**
 * Toolbar action that discovers Pandas DataFrames in the current debugger scope
 * and opens the time-series plot dialog.
 *
 * The action is enabled only when:
 *  - A Python debug session is active, AND
 *  - The session is currently paused at a breakpoint.
 */
class ShowPandasPlotAction : AnAction() {

    override fun update(e: AnActionEvent) {
        val project = e.project
        if (project == null) {
            e.presentation.isEnabledAndVisible = false
            return
        }

        val session = XDebuggerManager.getInstance(project).currentSession
        // Enable only when the debugger is paused (not running)
        e.presentation.isEnabled = session != null && session.isPaused
        e.presentation.isVisible = true
    }

    override fun actionPerformed(e: AnActionEvent) {
        val project = e.project ?: return

        val session = XDebuggerManager.getInstance(project).currentSession
        if (session == null || !session.isPaused) {
            Messages.showInfoMessage(
                project,
                "The debugger is not currently paused at a breakpoint.",
                "Pandas Plot"
            )
            return
        }

        // Run discovery on a background thread to avoid blocking the EDT
        ProgressManager.getInstance().run(
            object : Task.Backgroundable(project, "Scanning for Pandas DataFrames…", false) {

                private var dataFrames: List<DataFrameInfo> = emptyList()

                override fun run(indicator: ProgressIndicator) {
                    indicator.isIndeterminate = true
                    val evaluator = DataFrameEvaluator(project)
                    dataFrames = evaluator.findDataFrames()
                }

                override fun onSuccess() {
                    ApplicationManager.getApplication().invokeLater {
                        if (dataFrames.isEmpty()) {
                            Messages.showInfoMessage(
                                project,
                                "No Pandas DataFrames were found in the current scope.\n\n" +
                                    "Make sure you are paused inside a function or scope that\n" +
                                    "contains a variable of type pandas.DataFrame.",
                                "Pandas Plot – No DataFrames Found"
                            )
                            return@invokeLater
                        }

                        val dialog = PandasPlotDialog(project, dataFrames)
                        dialog.show()
                    }
                }

                override fun onThrowable(error: Throwable) {
                    ApplicationManager.getApplication().invokeLater {
                        Messages.showErrorDialog(
                            project,
                            "Failed to scan for DataFrames:\n${error.message}",
                            "Pandas Plot Error"
                        )
                    }
                }
            }
        )
    }
}
