package ksw.potentiak.justadogidentificationservice

import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel

class JDISViewModel : ViewModel() {
    private val _probabilities = MutableLiveData<List<DogClassificator>>()
    val probabilities: LiveData<List<DogClassificator>> = _probabilities
    fun updateProbabilities(dogClassificator: List<DogClassificator>){
        _probabilities.postValue(dogClassificator)
    }
}
data class DogClassificator(val label:String, val confidence:Float) {
    override fun toString():String{ return "$label / $probabilityString" }
    val probabilityString = String.format("%.1f%%", confidence * 100.0f)
}