package com.pandasplot.debugger

import com.intellij.openapi.diagnostic.logger
import com.intellij.openapi.editor.colors.TextAttributesKey
import com.intellij.openapi.project.Project
import com.intellij.xdebugger.XDebuggerManager
import com.intellij.xdebugger.evaluation.XDebuggerEvaluator
import com.intellij.xdebugger.frame.XValue
import com.intellij.xdebugger.frame.XValueNode
import com.intellij.xdebugger.frame.XValuePlace
import org.json.JSONArray
import org.json.JSONObject
import java.util.concurrent.CompletableFuture
import java.util.concurrent.TimeUnit
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
     */
    fun findDataFrames(): List<DataFrameInfo> {
        // Step 1: get all local variable names
        val namesJson = evalSync(
            """
            __import__('json').dumps(
                [k for k, v in locals().items()
                 if not k.startswith('_')
                 and type(v).__name__ == 'DataFrame'
                 and hasattr(v, 'columns')]
            )
            """.trimIndent()
        ) ?: return emptyList()

        val names = try {
            JSONArray(stripPythonStringQuotes(namesJson))
        } catch (e: Exception) {
            LOG.warn("Could not parse DataFrame names JSON: $namesJson", e)
            return emptyList()
        }

        return (0 until names.length()).mapNotNull { i ->
            val name = names.getString(i)
            loadDataFrameInfo(name)
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
     */
    private fun evalSync(expression: String, timeoutMs: Long = 10_000): String? {
        val session = XDebuggerManager.getInstance(project).currentSession ?: return null
        val frame = session.currentStackFrame ?: return null
        val evaluator = frame.evaluator ?: return null

        val future = CompletableFuture<String?>()

        evaluator.evaluate(expression, object : XDebuggerEvaluator.XEvaluationCallback {
            override fun evaluated(result: XValue) {
                result.computePresentation(object : XValueNode {
                    override fun setPresentation(
                        icon: Icon?,
                        type: String?,
                        value: String,
                        hasChildren: Boolean
                    ) {
                        future.complete(value)
                    }

                    // Remaining XValueNode methods – unused but required by interface
                    override fun setPresentation(
                        icon: Icon?,
                        presentation: com.intellij.xdebugger.frame.presentation.XValuePresentation,
                        hasChildren: Boolean
                    ) {
                        // Render via the presentation's renderValue
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
                        if (sb.isNotEmpty()) future.complete(sb.toString())
                        else future.complete(null)
                    }

                    override fun setFullValueEvaluator(fullValueEvaluator: com.intellij.xdebugger.frame.XFullValueEvaluator) {}
                    override fun isObsolete(): Boolean = future.isDone
                }, XValuePlace.TOOLTIP)
            }

            override fun errorOccurred(errorMessage: String) {
                LOG.debug("Evaluation error for expression [$expression]: $errorMessage")
                future.complete(null)
            }
        }, null)

        return try {
            future.get(timeoutMs, TimeUnit.MILLISECONDS)
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
