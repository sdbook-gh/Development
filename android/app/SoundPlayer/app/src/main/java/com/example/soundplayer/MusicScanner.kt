package com.example.soundplayer

import android.content.ContentResolver
import android.content.ContentUris
import android.net.Uri
import android.provider.MediaStore

/**
 * 通过 MediaStore 扫描外部存储中的 MP3 文件
 */
object MusicScanner {

    /**
     * 查询外部存储中的所有 MP3 文件
     *
     * @param resolver ContentResolver
     * @return 音乐列表，按标题排序
     */
    fun scan(resolver: ContentResolver): List<MusicItem> {
        val collection = MediaStore.Audio.Media.EXTERNAL_CONTENT_URI

        val projection = arrayOf(
            MediaStore.Audio.Media._ID,
            MediaStore.Audio.Media.TITLE,
            MediaStore.Audio.Media.ARTIST,
            MediaStore.Audio.Media.ALBUM,
            MediaStore.Audio.Media.DURATION
        )

        val selection = "${MediaStore.Audio.Media.MIME_TYPE} = ?"
        val selectionArgs = arrayOf("audio/mpeg")
        val sortOrder = "${MediaStore.Audio.Media.TITLE} ASC"

        val result = mutableListOf<MusicItem>()

        val cursor = resolver.query(
            collection,
            projection,
            selection,
            selectionArgs,
            sortOrder
        ) ?: return emptyList()

        cursor.use { c ->
            val idColumn = c.getColumnIndexOrThrow(MediaStore.Audio.Media._ID)
            val titleColumn = c.getColumnIndexOrThrow(MediaStore.Audio.Media.TITLE)
            val artistColumn = c.getColumnIndexOrThrow(MediaStore.Audio.Media.ARTIST)
            val albumColumn = c.getColumnIndexOrThrow(MediaStore.Audio.Media.ALBUM)
            val durationColumn = c.getColumnIndexOrThrow(MediaStore.Audio.Media.DURATION)

            while (c.moveToNext()) {
                val id = c.getLong(idColumn)
                val title = c.getString(titleColumn) ?: "未知标题"
                val artist = c.getString(artistColumn) ?: "未知艺术家"
                val album = c.getString(albumColumn) ?: "未知专辑"
                val duration = c.getLong(durationColumn)
                val uri = ContentUris.withAppendedId(collection, id).toString()

                result.add(
                    MusicItem(
                        id = id,
                        title = title,
                        artist = artist,
                        album = album,
                        duration = duration,
                        uri = uri
                    )
                )
            }
        }

        return result
    }

    /**
     * 将毫秒时长格式化为 mm:ss
     */
    fun formatDuration(durationMs: Long): String {
        val totalSeconds = durationMs / 1000
        val minutes = totalSeconds / 60
        val seconds = totalSeconds % 60
        return String.format("%d:%02d", minutes, seconds)
    }
}
