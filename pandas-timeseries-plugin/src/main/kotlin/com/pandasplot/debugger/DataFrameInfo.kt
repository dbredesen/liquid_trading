package com.pandasplot.debugger

/** Sentinel value: use row number (0, 1, 2, …) as the X axis. */
const val X_AXIS_ROWNUM = "<rownum>"

/** Sentinel value: use the DataFrame's index as the X axis. */
const val X_AXIS_INDEX = "DataFrame index"

/**
 * Metadata about a Pandas DataFrame found in the debugger scope.
 */
data class DataFrameInfo(
    val name: String,
    val columns: List<String>,
    val dtypes: Map<String, String>,
    val rowCount: Int,
    val shape: Pair<Int, Int>,
    /** dtype string of the DataFrame's index (e.g. "datetime64[ns]", "int64"). */
    val indexDtype: String = "object"
) {
    /** True when the index dtype looks like a datetime/timestamp type. */
    val isIndexDateTime: Boolean
        get() = indexDtype.lowercase().let { d ->
            d.contains("datetime") || d.contains("timestamp") ||
                d.contains("date") || d.contains("period")
        }

    /** Columns whose dtype suggests they hold datetime/timestamp values. */
    val timestampColumns: List<String>
        get() = columns.filter { col ->
            val dtype = dtypes[col]?.lowercase() ?: ""
            dtype.contains("datetime") || dtype.contains("timestamp") ||
                dtype.contains("date") || dtype.contains("period")
        }

    /** Columns whose dtype suggests they hold numeric values suitable for Y axis. */
    val numericColumns: List<String>
        get() = columns.filter { col ->
            val dtype = dtypes[col]?.lowercase() ?: ""
            dtype.startsWith("int") || dtype.startsWith("float") ||
                dtype.startsWith("uint") || dtype.contains("decimal") ||
                dtype.contains("complex") || dtype == "bool"
        }
}

/**
 * Extracted time series data ready for plotting.
 */
data class TimeSeriesData(
    val dfName: String,
    val timestampColumn: String,
    /** ISO-8601 strings, numeric strings, or row-number strings */
    val timestamps: List<String>,
    /** Map of column name -> list of values (null = NaN/missing) */
    val series: Map<String, List<Double?>>
)
