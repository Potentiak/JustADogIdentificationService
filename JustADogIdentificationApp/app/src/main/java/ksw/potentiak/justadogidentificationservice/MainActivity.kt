package ksw.potentiak.justadogidentificationservice

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.*
import android.os.Bundle
import android.util.Log
import android.util.Size
import android.widget.Toast
import androidx.activity.viewModels
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.core.Camera
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.Observer
import androidx.recyclerview.widget.RecyclerView
import ksw.potentiak.justadogidentificationservice.ml.*
import org.tensorflow.lite.examples.classification.util.YuvToRgbConverter
import org.tensorflow.lite.support.image.TensorImage
import java.util.concurrent.Executors


// Constants
private const val MAX_RESULT_DISPLAY = 5 // Maximum number of prediction results displayed
private const val REQUEST_CODE_PERMISSIONS = 621 // Return code after asking for permission
private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA) // permission needed(s) fot the app to work
private const val inputImageWidth = 224  // As described by tflite metadata
private const val inputImageHeight = 224 // ^

// Listener for the result of the ImageAnalyzer
typealias CNNListener = (dogClassificator: List<DogClassificator>) -> Unit
private const val TAG = "JaDIS APP"

class MainActivity : AppCompatActivity() {
    // CameraX variables
    private lateinit var preview: Preview // for displaying a preview...
    private lateinit var imageAnalyzer: ImageAnalysis // For feeding it to tflite model
    private lateinit var camera: Camera
    private val cameraExecutor = Executors.newSingleThreadExecutor()

    // Views attachment
    private val resultRecyclerView by lazy {
        findViewById<RecyclerView>(R.id.cnn_output_view) // Display the result of analysis
    }
    private val viewFinder by lazy {
        findViewById<PreviewView>(R.id.camera_preview) // Display the preview image from Camera
    }
    private val viewModel: JDISViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Camera permission handling
        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(
                this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS
            )
        }

        // Recycler view with model results
        val viewAdapter = RecognitionAdapter(this)
        resultRecyclerView.adapter = viewAdapter
        resultRecyclerView.itemAnimator = null
        viewModel.probabilities.observe(this,
            Observer {
                viewAdapter.submitList(it)
            }
        )

    }
    
    // permissions
    private fun allPermissionsGranted(): Boolean = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(
            baseContext, it
        ) == PackageManager.PERMISSION_GRANTED
    }
    
    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                startCamera()
            } else {
                Toast.makeText(
                    this,
                    getString(R.string.no_permission_text),
                    Toast.LENGTH_SHORT
                ).show()
                finish()
            }
        }
    }

    // Camera handling
    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener(Runnable {
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            preview = Preview.Builder()
                .build()

            imageAnalyzer = ImageAnalysis.Builder()
                .setTargetResolution(Size(inputImageWidth, inputImageHeight)) // so it is acceptable by tflite model
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also { analysisUseCase: ImageAnalysis ->
                    analysisUseCase.setAnalyzer(cameraExecutor, ImageAnalyzer(this) { items ->
                        // Finally, update the list of predictions based on tflite model output
                        viewModel.updateProbabilities(items)
                    })
                }

            // Select camera on the device
            val cameraSelector =
                if (cameraProvider.hasCamera(CameraSelector.DEFAULT_BACK_CAMERA))
                    CameraSelector.DEFAULT_BACK_CAMERA else CameraSelector.DEFAULT_FRONT_CAMERA
            try {
                cameraProvider.unbindAll() // To avoid binding camera when one is already bound
                camera = cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageAnalyzer
                )
                // Attach the preview to the preview window
                preview.setSurfaceProvider(viewFinder.surfaceProvider)
            } catch (e: Exception) {
                Log.e(TAG, "Use case binding failed", e)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private class ImageAnalyzer(ctx: Context, private val listener: CNNListener) :
        ImageAnalysis.Analyzer {
        private val jdisModel: LupusOmniDieMetadata by lazy{
            LupusOmniDieMetadata.newInstance(ctx)
        }
        // Function handling model communications with the app
        override fun analyze(imageProxy: ImageProxy) {
            val items = mutableListOf<DogClassificator>()
            val tfImage = TensorImage.fromBitmap(toBitmap(imageProxy)!!)
            val outputs = jdisModel.process(tfImage)
                .probabilityAsCategoryList.apply {
                    sortByDescending { it.score } // Sort with highest prediction first
                }.take(MAX_RESULT_DISPLAY) // Filter N top results
            for (output in outputs) {
                items.add(DogClassificator(output.label, output.score))
            }
            listener(items.toList())
            imageProxy.close()
        }
        
        private val yuvToRgbConverter = YuvToRgbConverter(ctx) // Using this method from
        // https://github.com/hoitab/TFLClassify because as of writing the app there is
        // no internal kotlin method for this
        private lateinit var bitmapBuffer: Bitmap
        private lateinit var rotationMatrix: Matrix

        @SuppressLint("UnsafeExperimentalUsageError", "UnsafeOptInUsageError")
        private fun toBitmap(imageProxy: ImageProxy): Bitmap? {

            val image = imageProxy.image ?: return null

            if (!::bitmapBuffer.isInitialized) {
                // The image rotation and RGB image buffer are initialized only once
                Log.d(TAG, "Initalise toBitmap()")
                rotationMatrix = Matrix()
                rotationMatrix.postRotate(imageProxy.imageInfo.rotationDegrees.toFloat())
                bitmapBuffer = Bitmap.createBitmap(
                    imageProxy.width, imageProxy.height, Bitmap.Config.ARGB_8888
                )
            }
            // Pass image to an image analyser
            yuvToRgbConverter.yuvToRgb(image, bitmapBuffer)
            // Create the Bitmap in the correct orientation
            return Bitmap.createBitmap(
                bitmapBuffer,
                0,
                0,
                bitmapBuffer.width,
                bitmapBuffer.height,
                rotationMatrix,
                false
            )
        }
    }
}
