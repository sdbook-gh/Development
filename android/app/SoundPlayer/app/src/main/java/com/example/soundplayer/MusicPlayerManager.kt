package com.example.soundplayer

import android.content.Context
import android.media.AudioAttributes
import android.media.AudioManager
import android.media.MediaPlayer
import android.net.Uri
import android.util.Log

/**
 * MediaPlayer 封装：支持循环播放、播放/暂停/切换、音量控制
 */
class MusicPlayerManager(private val context: Context) {

    companion object {
        private const val TAG = "MusicPlayerManager"
    }

    private var mediaPlayer: MediaPlayer? = null
    private var currentIndex: Int = -1
    private var playList: List<MusicItem> = emptyList()

    /** 快进/快退秒数（可通过设置修改） */
    var seekDurationSeconds: Int = 10

    /** 当前播放回调 */
    var onPlayingChanged: ((Boolean) -> Unit)? = null
    /** 切歌回调 */
    var onTrackChanged: ((Int, MusicItem) -> Unit)? = null

    /** 绑定播放列表 */
    fun setPlayList(list: List<MusicItem>) {
        playList = list
    }

    /** 当前播放索引 */
    fun getCurrentIndex(): Int = currentIndex

    /** 是否正在播放 */
    fun isPlaying(): Boolean = mediaPlayer?.isPlaying == true

    /**
     * 播放指定索引的歌曲（循环模式）
     */
    fun play(index: Int) {
        if (index < 0 || index >= playList.size) return

        val item = playList[index]
        currentIndex = index

        releasePlayer()

        try {
            mediaPlayer = MediaPlayer().apply {
                setAudioAttributes(
                    AudioAttributes.Builder()
                        .setUsage(AudioAttributes.USAGE_MEDIA)
                        .setContentType(AudioAttributes.CONTENT_TYPE_MUSIC)
                        .build()
                )
                setDataSource(context, Uri.parse(item.uri))
                isLooping = true  // 循环播放
                prepare()
                start()
                setOnCompletionListener {
                    // 理论上 looping=true 不会触发，但保留安全处理
                }
            }
            onTrackChanged?.invoke(index, item)
            onPlayingChanged?.invoke(true)
        } catch (e: Exception) {
            Log.e(TAG, "播放失败: ${e.message}", e)
        }
    }

    /** 播放/暂停切换 */
    fun togglePlayPause() {
        val mp = mediaPlayer ?: return
        if (mp.isPlaying) {
            mp.pause()
            onPlayingChanged?.invoke(false)
        } else {
            mp.start()
            onPlayingChanged?.invoke(true)
        }
    }

    /** 下一首 */
    fun next() {
        if (playList.isEmpty()) return
        val nextIndex = (currentIndex + 1) % playList.size
        play(nextIndex)
    }

    /** 上一首 */
    fun previous() {
        if (playList.isEmpty()) return
        val prevIndex = if (currentIndex <= 0) playList.size - 1 else currentIndex - 1
        play(prevIndex)
    }

    // ======================== 快进/快退 ========================

    /** 快退 seekDurationSeconds 秒 */
    fun seekBackward() {
        val mp = mediaPlayer ?: return
        val newPos = (mp.currentPosition - seekDurationSeconds * 1000).coerceAtLeast(0)
        mp.seekTo(newPos)
    }

    /** 快进 seekDurationSeconds 秒 */
    fun seekForward() {
        val mp = mediaPlayer ?: return
        val duration = if (mp.duration > 0) mp.duration else Int.MAX_VALUE
        val newPos = (mp.currentPosition + seekDurationSeconds * 1000).coerceAtMost(duration)
        mp.seekTo(newPos)
    }

    // ======================== 音量控制 ========================

    /**
     * 获取系统媒体音量 (0~maxVolume)
     */
    fun getMaxVolume(): Int {
        val audioManager = context.getSystemService(Context.AUDIO_SERVICE) as AudioManager
        return audioManager.getStreamMaxVolume(AudioManager.STREAM_MUSIC)
    }

    fun getCurrentVolume(): Int {
        val audioManager = context.getSystemService(Context.AUDIO_SERVICE) as AudioManager
        return audioManager.getStreamVolume(AudioManager.STREAM_MUSIC)
    }

    /**
     * 设置系统媒体音量
     */
    fun setVolume(volume: Int) {
        val audioManager = context.getSystemService(Context.AUDIO_SERVICE) as AudioManager
        val max = audioManager.getStreamMaxVolume(AudioManager.STREAM_MUSIC)
        val clamped = volume.coerceIn(0, max)
        audioManager.setStreamVolume(
            AudioManager.STREAM_MUSIC,
            clamped,
            AudioManager.FLAG_SHOW_UI
        )
    }

    // ======================== 生命周期 ========================

    fun releasePlayer() {
        mediaPlayer?.let { mp ->
            if (mp.isPlaying) {
                mp.stop()
            }
            mp.release()
        }
        mediaPlayer = null
    }

    fun release() {
        releasePlayer()
        onPlayingChanged = null
        onTrackChanged = null
    }
}
