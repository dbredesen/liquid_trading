package com.pandasplot.debugger

/**
 * Metadata about a Pandas DataFrame found in the debugger scope.
 */
data class DataFrameInfo(
    val name: String,
    val columns: List<String>,
    val dtypes: Map<String, String>,
    val rowCount: Int,
    val shape: Pair<Int, Int>
) {
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
    /** ISO-8601 strings or numeric epoch values */
    val timestamps: List<String>,
    /** Map of column name -> list of values (null = NaN/missing) */
    val series: Map<String, List<Double?>>
)
