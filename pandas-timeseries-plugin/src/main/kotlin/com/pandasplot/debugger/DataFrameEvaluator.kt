package com.pandasplot.debugger

import com.intellij.openapi.diagnostic.logger
import com.intellij.openapi.editor.colors.TextAttributesKey
import com.intellij.openapi.project.Project
import com.intellij.ui.SimpleTextAttributes
import com.intellij.xdebugger.XDebuggerManager
import com.intellij.xdebugger.evaluation.XDebuggerEvaluator
import com.intellij.xdebugger.frame.XCompositeNode
import com.intellij.xdebugger.frame.XDebuggerTreeNodeHyperlink
import com.intellij.xdebugger.frame.XStackFrame
import com.intellij.xdebugger.frame.XValue
import com.intellij.xdebugger.frame.XValueChildrenList
import com.intellij.xdebugger.frame.XValueNode
import com.intellij.xdebugger.frame.XValuePlace
import com.intellij.xdebugger.frame.presentation.XValuePresentation
import org.json.JSONArray
import org.json.JSONObject
import java.awt.Font
import java.util.concurrent.CompletableFuture
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import javax.swing.Icon

private val LOG = logger<DataFrameEvaluator>()

/**
 * Evaluates Python expressions in the current PyCharm debugger session to discover
 * Pandas DataFrames in scope and extract their data for plotting.
 */
class DataFrameEvaluator(private val project: Project) {

    // ---------------------------------------------------------------------------
    // Public API
    // ---------------------------------------------------------------------------

    /**
     * Returns all Pandas DataFrames found in the current debugger frame's local scope.
     * Must be called from a background thread (does a blocking wait on the evaluator).
     *
     * Uses [XStackFrame.computeChildren] to enumerate variables directly from the
     * debugger API instead of evaluating `locals()` inside a Python expression.
     * This is necessary because PyCharm 2025.3's pydevd evaluates expressions in a
     * sub-scope where `locals()` no longer reflects the stopped frame's variables.
     */
    fun findDataFrames(): List<DataFrameInfo> {
        val session = XDebuggerManager.getInstance(project).currentSession ?: return emptyList()
        val frame = session.currentStackFrame ?: return emptyList()
        val dfNames = collectDataFrameNames(frame)
        return dfNames.mapNotNull { loadDataFrameInfo(it) }
    }

    /**
     * Walks the frame's variable tree and returns the names of all variables whose
     * presentation type contains "DataFrame".
     */
    private fun collectDataFrameNames(frame: XStackFrame): List<String> {
        val doneFuture = CompletableFuture<List<String>>()
        // CopyOnWriteArrayList because addChildren + computePresentation callbacks
        // may arrive on different threads.
        val pending = CopyOnWriteArrayList<Pair<String, CompletableFuture<Boolean>>>()

        frame.computeChildren(object : XCompositeNode {
            override fun addChildren(children: XValueChildrenList, last: Boolean) {
                for (i in 0 until children.size()) {
                    val name = children.getName(i) ?: continue
                    if (name.startsWith("_")) continue

                    val isDF = CompletableFuture<Boolean>()
                    pending.add(name to isDF)

                    children.getValue(i).computePresentation(object : XValueNode {
                        private fun resolve(type: String?, renderedValue: String) {
                            isDF.complete(
                                (type != null && type.contains("DataFrame")) ||
                                renderedValue.contains("DataFrame")
                            )
                        }

                        override fun setPresentation(
                            icon: Icon?, type: String?, value: String, hasChildren: Boolean
                        ) = resolve(type, value)

                        override fun setPresentation(
                            icon: Icon?, presentation: XValuePresentation, hasChildren: Boolean
                        ) {
                            val sb = StringBuilder()
                            presentation.renderValue(object : XValuePresentation.XValueTextRenderer {
                                override fun renderValue(value: String) { sb.append(value) }
                                override fun renderValue(value: String, key: TextAttributesKey) { sb.append(value) }
                                override fun renderStringValue(value: String) { sb.append(value) }
                                override fun renderNumericValue(value: String) { sb.append(value) }
                                override fun renderKeywordValue(value: String) { sb.append(value) }
                                override fun renderComment(comment: String) {}
                                override fun renderError(error: String) {}
                                override fun renderSpecialSymbol(symbol: String) { sb.append(symbol) }
                                override fun renderStringValue(
                                    value: String, additionalSpecialCharsToHighlight: String?, maxLength: Int
                                ) { sb.append(value) }
                            })
                            resolve(presentation.type, sb.toString())
                        }

                        override fun setFullValueEvaluator(e: com.intellij.xdebugger.frame.XFullValueEvaluator) {}
                        override fun isObsolete(): Boolean = isDF.isDone
                    }, XValuePlace.TOOLTIP)
                }

                if (last) {
                    val names = pending.mapNotNull { (name, future) ->
                        try {
                            if (future.get(5_000, TimeUnit.MILLISECONDS) == true) name else null
                        } catch (e: Exception) {
                            LOG.warn("Timeout resolving type for variable '$name'", e)
                            null
                        }
                    }
                    doneFuture.complete(names)
                }
            }

            override fun tooManyChildren(remaining: Int) { /* accept the first batch */ }
            override fun setAlreadySorted(alreadySorted: Boolean) {}
            override fun setMessage(msg: String, icon: Icon?, attrs: SimpleTextAttributes, link: XDebuggerTreeNodeHyperlink?) {}
            override fun setErrorMessage(errorMessage: String) {
                LOG.warn("computeChildren error: $errorMessage")
                doneFuture.complete(emptyList())
            }
            override fun setErrorMessage(errorMessage: String, link: XDebuggerTreeNodeHyperlink?) {
                LOG.warn("computeChildren error: $errorMessage")
                doneFuture.complete(emptyList())
            }
            override fun isObsolete(): Boolean = doneFuture.isDone
        })

        return try {
            doneFuture.get(15_000, TimeUnit.MILLISECONDS)
        } catch (e: Exception) {
            LOG.warn("Timeout waiting for frame children: ${e.message}")
            emptyList()
        }
    }

    /**
     * Extracts time series data from [dfName] for the given columns.
     * [timestampCol] will become the X axis; [seriesCols] become Y axis series.
     */
    fun extractTimeSeriesData(
        dfName: String,
        timestampCol: String,
        seriesCols: List<String>
    ): TimeSeriesData? {
        if (seriesCols.isEmpty()) return null

        // Build a list of columns to extract, escaping names for safety
        val allCols = (listOf(timestampCol) + seriesCols).joinToString(",") { "\"${it.escapeForPython()}\"" }

        val script = """
            (lambda df, cols: __import__('json').dumps({
                'timestamps': df[cols[0]].astype(str).tolist(),
                'series': {c: [None if __import__('math').isnan(float(x)) else float(x)
                               for x in df[c].fillna(float('nan'))]
                           for c in cols[1:]}
            }))(${dfName}, [${allCols}])
        """.trimIndent()

        val raw = evalSync(script) ?: return null

        return try {
            val json = JSONObject(stripPythonStringQuotes(raw))
            val tsArray = json.getJSONArray("timestamps")
            val timestamps = (0 until tsArray.length()).map { tsArray.getString(it) }

            val seriesObj = json.getJSONObject("series")
            val seriesMap = seriesCols.associateWith { col ->
                val arr = seriesObj.optJSONArray(col) ?: return@associateWith emptyList<Double?>()
                (0 until arr.length()).map { if (arr.isNull(it)) null else arr.getDouble(it) }
            }

            TimeSeriesData(
                dfName = dfName,
                timestampColumn = timestampCol,
                timestamps = timestamps,
                series = seriesMap
            )
        } catch (e: Exception) {
            LOG.warn("Failed to parse time series JSON for $dfName", e)
            null
        }
    }

    // ---------------------------------------------------------------------------
    // Private helpers
    // ---------------------------------------------------------------------------

    private fun loadDataFrameInfo(name: String): DataFrameInfo? {
        val script = """
            (lambda df: __import__('json').dumps({
                'columns': df.columns.tolist(),
                'dtypes': {c: str(t) for c, t in df.dtypes.items()},
                'rows': len(df),
                'shape': list(df.shape)
            }))(${name})
        """.trimIndent()

        val raw = evalSync(script) ?: return null

        return try {
            val json = JSONObject(stripPythonStringQuotes(raw))
            val colArray = json.getJSONArray("columns")
            val columns = (0 until colArray.length()).map { colArray.getString(it) }

            val dtypesObj = json.getJSONObject("dtypes")
            val dtypes = columns.associateWith { dtypesObj.optString(it, "object") }

            val shapeArray = json.getJSONArray("shape")

            DataFrameInfo(
                name = name,
                columns = columns,
                dtypes = dtypes,
                rowCount = json.getInt("rows"),
                shape = Pair(shapeArray.getInt(0), shapeArray.getInt(1))
            )
        } catch (e: Exception) {
            LOG.warn("Failed to parse DataFrame info for $name: $raw", e)
            null
        }
    }

    /**
     * Evaluates [expression] in the current debug frame and blocks until the result
     * is available (up to [timeoutMs] ms). Returns the raw string representation
     * of the evaluated value, or null on failure / timeout.
     *
     * pydevd truncates long string values in [XValueNode.setPresentation]. When that
     * happens it also calls [XValueNode.setFullValueEvaluator] with an evaluator that
     * can fetch the complete string. We honour that evaluator so that large JSON payloads
     * (e.g. DataFrames with hundreds of columns) are returned untruncated.
     */
    private fun evalSync(expression: String, timeoutMs: Long = 10_000): String? {
        val session = XDebuggerManager.getInstance(project).currentSession ?: return null
        val frame = session.currentStackFrame ?: return null
        val evaluator = frame.evaluator ?: return null

        val presentationFuture = CompletableFuture<String?>()
        val fullValueFuture    = CompletableFuture<String?>()
        val fullEvalStarted    = AtomicBoolean(false)

        evaluator.evaluate(expression, object : XDebuggerEvaluator.XEvaluationCallback {
            override fun evaluated(result: XValue) {
                result.computePresentation(object : XValueNode {
                    override fun setPresentation(icon: Icon?, type: String?, value: String, hasChildren: Boolean) {
                        presentationFuture.complete(value)
                    }

                    override fun setPresentation(
                        icon: Icon?,
                        presentation: com.intellij.xdebugger.frame.presentation.XValuePresentation,
                        hasChildren: Boolean
                    ) {
                        val sb = StringBuilder()
                        presentation.renderValue(object :
                            com.intellij.xdebugger.frame.presentation.XValuePresentation.XValueTextRenderer {
                            override fun renderValue(value: String) { sb.append(value) }
                            override fun renderValue(value: String, textAttributes: TextAttributesKey) { sb.append(value) }
                            override fun renderStringValue(value: String) { sb.append(value) }
                            override fun renderNumericValue(value: String) { sb.append(value) }
                            override fun renderKeywordValue(value: String) { sb.append(value) }
                            override fun renderComment(comment: String) {}
                            override fun renderError(error: String) { sb.append(error) }
                            override fun renderSpecialSymbol(symbol: String) { sb.append(symbol) }
                            override fun renderStringValue(
                                value: String, additionalSpecialCharsToHighlight: String?, maxLength: Int
                            ) { sb.append(value) }
                        })
                        presentationFuture.complete(if (sb.isNotEmpty()) sb.toString() else null)
                    }

                    // Called by pydevd immediately after setPresentation when the displayed
                    // value was truncated.  Start the full-value fetch so we can return
                    // untruncated JSON to the caller.
                    override fun setFullValueEvaluator(fve: com.intellij.xdebugger.frame.XFullValueEvaluator) {
                        fullEvalStarted.set(true)
                        fve.startEvaluation(object :
                            com.intellij.xdebugger.frame.XFullValueEvaluator.XFullValueEvaluationCallback {
                            override fun evaluated(fullValue: String) { fullValueFuture.complete(fullValue) }
                            override fun evaluated(fullValue: String, font: Font?) { fullValueFuture.complete(fullValue) }
                            override fun errorOccurred(msg: String)   { fullValueFuture.complete(null) }
                        })
                    }

                    override fun isObsolete(): Boolean = presentationFuture.isDone
                }, XValuePlace.TOOLTIP)
            }

            override fun errorOccurred(errorMessage: String) {
                LOG.debug("Evaluation error for expression [$expression]: $errorMessage")
                presentationFuture.complete(null)
                fullValueFuture.complete(null)
            }
        }, null)

        return try {
            val presented = presentationFuture.get(timeoutMs, TimeUnit.MILLISECONDS)
            // setFullValueEvaluator is called synchronously on the same callback thread
            // right after setPresentation returns.  A brief pause ensures fullEvalStarted
            // is visible before we decide whether to wait for the full value.
            Thread.sleep(50)
            when {
                fullValueFuture.isDone ->
                    // Full value already available (fast path).
                    fullValueFuture.getNow(null) ?: presented
                fullEvalStarted.get() ->
                    // Full-value fetch is in-flight; wait for it.
                    try {
                        fullValueFuture.get(timeoutMs, TimeUnit.MILLISECONDS) ?: presented
                    } catch (e: Exception) {
                        LOG.warn("Full-value fetch timed out; using truncated presentation")
                        presented
                    }
                else ->
                    // Value was not truncated; use the presentation directly.
                    presented
            }
        } catch (e: Exception) {
            LOG.warn("Evaluation timed out or failed: ${e.message}")
            null
        }
    }

    /**
     * Python string representations are wrapped in single or double quotes.
     * Strip them to get the raw JSON content.
     */
    private fun stripPythonStringQuotes(s: String): String {
        val trimmed = s.trim()
        return when {
            trimmed.startsWith("'") && trimmed.endsWith("'") ->
                trimmed.substring(1, trimmed.length - 1).replace("\\'", "'")
            trimmed.startsWith("\"") && trimmed.endsWith("\"") ->
                trimmed.substring(1, trimmed.length - 1).replace("\\\"", "\"")
            else -> trimmed
        }
    }

    private fun String.escapeForPython(): String =
        this.replace("\\", "\\\\").replace("\"", "\\\"")
}
