package com.example.soundplayer

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView

/**
 * 音乐列表适配器
 */
class MusicAdapter(
    private val onItemClick: (Int) -> Unit
) : RecyclerView.Adapter<MusicAdapter.MusicViewHolder>() {

    private val items = mutableListOf<MusicItem>()
    private var currentPlayingIndex: Int = -1

    /** 更新数据 */
    fun submitList(list: List<MusicItem>) {
        items.clear()
        items.addAll(list)
        currentPlayingIndex = -1
        notifyDataSetChanged()
    }

    /** 设置当前播放高亮项 */
    fun setCurrentPlaying(index: Int) {
        val oldIndex = currentPlayingIndex
        currentPlayingIndex = index
        if (oldIndex >= 0) {
            notifyItemChanged(oldIndex)
        }
        if (index >= 0 && index < items.size) {
            notifyItemChanged(index)
        }
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): MusicViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_music, parent, false)
        return MusicViewHolder(view)
    }

    override fun onBindViewHolder(holder: MusicViewHolder, position: Int) {
        holder.bind(items[position], position == currentPlayingIndex)
    }

    override fun getItemCount(): Int = items.size

    inner class MusicViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        private val titleText: TextView = itemView.findViewById(R.id.tv_title)
        private val artistText: TextView = itemView.findViewById(R.id.tv_artist)
        private val durationText: TextView = itemView.findViewById(R.id.tv_duration)

        init {
            itemView.setOnClickListener {
                val pos = bindingAdapterPosition
                if (pos != RecyclerView.NO_POSITION) {
                    onItemClick(pos)
                }
            }
        }

        fun bind(item: MusicItem, isPlaying: Boolean) {
            val prefix = if (isPlaying) "▶ " else "♪ "
            titleText.text = prefix + item.title
            artistText.text = "${item.artist} · ${item.album}"
            durationText.text = MusicScanner.formatDuration(item.duration)

            // 高亮正在播放的项
            itemView.setBackgroundColor(
                if (isPlaying) {
                    0x332196F3.toInt()  // 半透明蓝色
                } else {
                    0x00000000  // 透明
                }
            )
        }
    }
}
