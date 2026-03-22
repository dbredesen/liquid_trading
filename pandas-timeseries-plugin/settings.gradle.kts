plugins {
    // Auto-provisions any JDK version declared via kotlin { jvmToolchain(...) }
    // without requiring a manual brew install. Downloads from Foojay Adoptium.
    id("org.gradle.toolchains.foojay-resolver-convention") version "0.8.0"
}

rootProject.name = "pandas-timeseries-plugin"
