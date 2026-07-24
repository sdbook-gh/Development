package com.example.soundplayer

/**
 * 音乐文件数据模型
 */
data class MusicItem(
    val id: Long,
    val title: String,
    val artist: String,
    val album: String,
    val duration: Long,   // 毫秒
    val uri: String       // content:// 或 file:// URI
)
