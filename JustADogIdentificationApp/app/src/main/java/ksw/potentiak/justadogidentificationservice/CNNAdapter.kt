package ksw.potentiak.justadogidentificationservice

import android.content.Context
import android.view.LayoutInflater
import android.view.ViewGroup
import androidx.recyclerview.widget.DiffUtil
import androidx.recyclerview.widget.ListAdapter
import androidx.recyclerview.widget.RecyclerView
import ksw.potentiak.justadogidentificationservice.databinding.PredictionLabelBinding

class RecognitionAdapter(private val ctx: Context) :
    ListAdapter<DogClassificator, RecognitionViewHolder>(RecognitionDiffUtil()) {

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): RecognitionViewHolder {
        val inflater = LayoutInflater.from(ctx)
        val binding = PredictionLabelBinding.inflate(inflater, parent, false)
        return RecognitionViewHolder(binding)
    }
    // Binding the data fields to the RecognitionViewHolder
    override fun onBindViewHolder(holder: RecognitionViewHolder, position: Int) {
        holder.bindTo(getItem(position))
    }
    private class RecognitionDiffUtil : DiffUtil.ItemCallback<DogClassificator>() {
        override fun areItemsTheSame(oldItem: DogClassificator, newItem: DogClassificator): Boolean {
            return oldItem.label == newItem.label
        }
        override fun areContentsTheSame(oldItem: DogClassificator, newItem: DogClassificator): Boolean {
            return oldItem.confidence == newItem.confidence
        }
    }
}
class RecognitionViewHolder(private val binding: PredictionLabelBinding) :
    RecyclerView.ViewHolder(binding.root) {
    // Binding received probabilities from the model via viewbinding
    fun bindTo(dogClassificator: DogClassificator) {
        binding.predictionEntry = dogClassificator
        binding.executePendingBindings()
    }
}