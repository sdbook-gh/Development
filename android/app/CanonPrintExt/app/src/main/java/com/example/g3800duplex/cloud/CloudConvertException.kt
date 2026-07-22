package com.example.g3800duplex.cloud

class CloudConvertException(
    message: String,
    cause: Throwable? = null,
    val stage: String = "convert",
) : Exception(message, cause) {
    init {
        CloudLog.e(stage, message, cause)
    }
}
