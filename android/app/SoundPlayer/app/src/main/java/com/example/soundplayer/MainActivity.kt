package com.example.soundplayer

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.view.Menu
import android.view.MenuItem
import android.widget.SeekBar
import android.widget.Toast
import android.widget.ImageButton
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.google.android.material.appbar.MaterialToolbar
import com.google.android.material.floatingactionbutton.FloatingActionButton
import com.google.android.material.textview.MaterialTextView

class MainActivity : AppCompatActivity() {

    private lateinit var recyclerView: RecyclerView
    private lateinit var adapter: MusicAdapter
    private lateinit var playerManager: MusicPlayerManager
    private lateinit var toolbar: MaterialToolbar

    // 底部控制栏
    private lateinit var tvCurrentTitle: MaterialTextView
    private lateinit var btnRewind: ImageButton
    private lateinit var btnPrev: ImageButton
    private lateinit var btnPlayPause: FloatingActionButton
    private lateinit var btnNext: ImageButton
    private lateinit var btnForward: ImageButton
    private lateinit var volumeSeekBar: SeekBar

    companion object {
        private const val PREFS_NAME = "soundplayer_prefs"
        private const val KEY_SEEK_SECONDS = "seek_seconds"
        private const val DEFAULT_SEEK_SECONDS = 10
        // 菜单 ID 到秒数的映射
        private val SEEK_OPTIONS = linkedMapOf(
            R.id.seek_5 to 5,
            R.id.seek_10 to 10,
            R.id.seek_20 to 20,
            R.id.seek_30 to 30,
            R.id.seek_60 to 60
        )
    }

    // 权限请求
    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) {
            loadMusicList()
        } else {
            Toast.makeText(this, "需要存储权限才能扫描音乐文件", Toast.LENGTH_LONG).show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        initViews()
        initPlayerManager()
        requestPermission()
    }

    private fun initViews() {
        toolbar = findViewById(R.id.toolbar)
        setSupportActionBar(toolbar)

        recyclerView = findViewById(R.id.recycler_view)
        tvCurrentTitle = findViewById(R.id.tv_current_title)
        btnRewind = findViewById(R.id.btn_rewind)
        btnPrev = findViewById(R.id.btn_prev)
        btnPlayPause = findViewById(R.id.btn_play_pause)
        btnNext = findViewById(R.id.btn_next)
        btnForward = findViewById(R.id.btn_forward)
        volumeSeekBar = findViewById(R.id.seekbar_volume)

        adapter = MusicAdapter { position ->
            // 点击列表项 -> 播放该歌曲（循环模式）
            playerManager.play(position)
            adapter.setCurrentPlaying(position)
        }

        recyclerView.layoutManager = LinearLayoutManager(this)
        recyclerView.adapter = adapter

        // 播放控制按钮
        btnRewind.setOnClickListener { playerManager.seekBackward() }
        btnPrev.setOnClickListener { playerManager.previous() }
        btnPlayPause.setOnClickListener { playerManager.togglePlayPause() }
        btnNext.setOnClickListener { playerManager.next() }
        btnForward.setOnClickListener { playerManager.seekForward() }

        // 音量 SeekBar
        volumeSeekBar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                if (fromUser) {
                    playerManager.setVolume(progress)
                }
            }
            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })
    }

    private fun initPlayerManager() {
        playerManager = MusicPlayerManager(this)

        // 从 SharedPreferences 读取快进/快退秒数
        val prefs = getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        playerManager.seekDurationSeconds = prefs.getInt(KEY_SEEK_SECONDS, DEFAULT_SEEK_SECONDS)

        // 播放状态变化 -> 更新按钮图标
        playerManager.onPlayingChanged = { isPlaying ->
            btnPlayPause.setImageResource(
                if (isPlaying) R.drawable.ic_pause else R.drawable.ic_play
            )
        }

        // 切歌 -> 更新标题和高亮
        playerManager.onTrackChanged = { index, item ->
            tvCurrentTitle.text = "♪ ${item.title}"
            adapter.setCurrentPlaying(index)
        }
    }

    // ======================== 设置菜单 ========================

    override fun onCreateOptionsMenu(menu: Menu): Boolean {
        menuInflater.inflate(R.menu.main_menu, menu)

        // 勾选当前生效的秒数
        val currentSeconds = playerManager.seekDurationSeconds
        val menuItemId = SEEK_OPTIONS.entries.find { it.value == currentSeconds }?.key
        menuItemId?.let { menu.findItem(it)?.isChecked = true }

        return true
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        val seconds = SEEK_OPTIONS[item.itemId]
        if (seconds != null) {
            // 更新播放器
            playerManager.seekDurationSeconds = seconds
            // 持久化
            getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
                .edit()
                .putInt(KEY_SEEK_SECONDS, seconds)
                .apply()
            // 更新选中状态
            item.isChecked = true
            Toast.makeText(this, "快进/快退已设为 ${seconds} 秒", Toast.LENGTH_SHORT).show()
            return true
        }
        return super.onOptionsItemSelected(item)
    }

    // ======================== 权限 & 加载 ========================

    private fun requestPermission() {
        val permission = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            Manifest.permission.READ_MEDIA_AUDIO
        } else {
            Manifest.permission.READ_EXTERNAL_STORAGE
        }

        if (ContextCompat.checkSelfPermission(this, permission) == PackageManager.PERMISSION_GRANTED) {
            loadMusicList()
        } else {
            permissionLauncher.launch(permission)
        }
    }

    private fun loadMusicList() {
        val musicList = MusicScanner.scan(contentResolver)

        if (musicList.isEmpty()) {
            Toast.makeText(this, "未找到 MP3 文件", Toast.LENGTH_SHORT).show()
            return
        }

        playerManager.setPlayList(musicList)
        adapter.submitList(musicList)

        // 初始化音量 SeekBar
        volumeSeekBar.max = playerManager.getMaxVolume()
        volumeSeekBar.progress = playerManager.getCurrentVolume()

        Toast.makeText(this, "扫描到 ${musicList.size} 首歌曲", Toast.LENGTH_SHORT).show()
    }

    override fun onDestroy() {
        super.onDestroy()
        playerManager.release()
    }
}
