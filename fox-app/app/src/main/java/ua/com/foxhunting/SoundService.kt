package ua.com.foxhunting

import android.app.Service
import android.content.Context
import android.content.Intent
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.media.AudioRecord
import android.os.Binder
import android.os.IBinder
import android.util.Log
import android.widget.Toast
import com.google.firebase.firestore.FieldValue
import com.google.firebase.firestore.FirebaseFirestore
import org.tensorflow.lite.support.audio.TensorAudio
import org.tensorflow.lite.support.label.Category
import org.tensorflow.lite.task.audio.classifier.AudioClassifier
import java.text.DateFormat
import java.text.SimpleDateFormat
import java.util.*
import kotlin.concurrent.scheduleAtFixedRate

class SoundService : Service(), LocationListener {

    var PREF_NAME = "fox"
    var USER_ID_KEY = "user_id"
    var LATTITUDE_KEY = "latitude"
    var LONGITUDE_KEY = "longitude"

    var LogOut:String =""


    private var sampleRate: Int=0
    var TAG = "SoundService"

    // Binder given to clients
    private val binder = SoundBinder()

    private var allowRebind: Boolean = false

    var modelPath = "pushki_model.tflite"
    var probabilityThreshold: Float = 0.75f
    var probabilityThresholdOld: Float = 0.8f
    var silence: String = "Silence"
    var pushka: String = "pushka"
    lateinit var userId: String

    private lateinit var tensor: TensorAudio;
    private lateinit var record: AudioRecord;
    private lateinit var classifier: AudioClassifier;
    lateinit var pukEvents: ArrayList<PukEvent>;
    private lateinit var rawEvents: ArrayList<PukEvent>;
    lateinit var pukNotofocation: ArrayList<String>;

    var curLocation: Location?=null;
    var  curLatitude:Double?=null
    var  curLongitude:Double?=null
    lateinit var recorderSpecs:String
    private lateinit var locationManager: LocationManager
    private val locationPermissionCode = 2


    /**
     * Class used for the client Binder.  Because we know this service always
     * runs in the same process as its clients, we don't need to deal with IPC.
     */
    inner class SoundBinder : Binder() {
        // Return this instance of LocalService so clients can call public methods
        fun getService(): SoundService = this@SoundService
    }
    fun saveSetting(){
        var userSetting = getSharedPreferences(PREF_NAME,  MODE_PRIVATE);
        val editor =  userSetting.edit()
        editor.putString(USER_ID_KEY, userId)
        if(curLatitude!=null)
            editor.putFloat(LATTITUDE_KEY, curLatitude!!.toFloat())
        if(curLongitude!=null)
            editor.putFloat(LONGITUDE_KEY, curLongitude!!.toFloat())
        editor.apply()
        editor.commit()

    }
    override fun onCreate() {
        // The service is being created

        pukEvents = ArrayList()
        rawEvents = ArrayList()
        pukNotofocation = ArrayList()
        var userSetting = getSharedPreferences(PREF_NAME,  MODE_PRIVATE);
        userId = userSetting.getString(USER_ID_KEY, UUID.randomUUID().toString()).toString()
        curLatitude = userSetting.getFloat(LATTITUDE_KEY, 0F).toDouble()
        curLongitude = userSetting.getFloat(LONGITUDE_KEY, 0F).toDouble()
        saveSetting();
        getLocation();
        try{
        // TODO 2.3: Loading the model from the assets folder
        classifier = AudioClassifier.createFromFile(this, modelPath)

        // TODO 3.1: Creating an audio recorder
        tensor = classifier.createInputTensorAudio()

        // TODO 3.2: showing the audio recorder specification
        val format = classifier.requiredTensorAudioFormat
        recorderSpecs = "Number Of Channels: ${format.channels}\n" +
                "Sample Rate: ${format.sampleRate}"
        sampleRate= format.sampleRate

        // TODO 3.3: Creating
        record = classifier.createAudioRecord()
        record.startRecording()
    }catch (e: Exception) {
        Toast.makeText(
            applicationContext,
            "model. "+e.message,
            Toast.LENGTH_SHORT
        ).show()
        Log.i("ML", e.toString())
    }
        Timer().scheduleAtFixedRate(1, 500) {
            try{

                val numberOfSamples = tensor.load(record)
                val output = classifier.classify(tensor)

                // TODO 4.2: Filtering out classifications with low probability
                val filteredModelOutputOld = output[0].categories.filter {
                    it.score > probabilityThresholdOld
                }.sortedBy { -it.score }

                var modelOld: Category? =  null;
                if(filteredModelOutputOld.isNotEmpty())
                    modelOld = filteredModelOutputOld[0];

                val filteredModelOutput = output[1].categories.filter {
                    it.score > probabilityThreshold
                }.sortedBy { -it.score }
                var model: Category? =  null;
                if(filteredModelOutput.isNotEmpty())
                    model = filteredModelOutput[0];

                // TODO 4.3: Creating a multiline string with the filtered results
                var outputStrOld =
                    filteredModelOutputOld.sortedBy { -it.score }
                        .joinToString(separator = "\n") { "${it.label} -> ${it.score} " }
                var outputStr =
                    filteredModelOutput.sortedBy { -it.score }
                        .joinToString(separator = "\n") { "${it.label} -> ${it.score} " }

                // TODO 4.4: Updating the UI
                if (outputStr.isNotEmpty()) {
                    LogOut = outputStr +"\n"+ outputStrOld
                    /*
                    runOnUiThread {
                        tvOutput.text = outputStr + outputStrOld
                    }
                    */

                }
                if(modelOld==null || modelOld.label!=silence){
                    // Dont silence
                    if(model!=null && model.label==pushka){
                        val  currentTime = Date()

                        var event = PukEvent()
                        event.score = model.score;
                        if(curLatitude!=null) {
                            event.latitude = curLatitude
                            event.longitude = curLongitude
                        }
                        event.type = model.label;
                        event.userId = userId;
                        var buffer = tensor.tensorBuffer.floatArray
                        var maxIndex:Int =0
                        var maxValue:Float = 0F
                        for ((index, value) in buffer.withIndex()){
                            if( Math.abs(value) >maxValue){
                                maxValue = Math.abs(value)
                                maxIndex = index
                            }
                        }
                        event.maxIndex = maxIndex;
                        event.maxValue = maxValue;
                        // Add offset
                        var offSet = (buffer.size-1-maxIndex)*1000/sampleRate
                        currentTime.time = (currentTime.time-offSet).toLong()
                        event.time =currentTime


                        rawEvents.add(event)
                        sendToDb()
                        val df: DateFormat = SimpleDateFormat("HH:mm:ss")
                        val msg = df.format(event.time)+ " "+"%.3f".format(event.maxValue) + " "+" - ${model.label}"
                        pukNotofocation.add(msg)

                    }
                }
            }catch (e: Exception) {
                Log.i("Yamnet", e.toString());
            }
        }

    }

    override fun onBind(intent: Intent): IBinder {

        return binder
    }
    override fun onUnbind(intent: Intent): Boolean {
        // All clients have unbound with unbindService()
        return allowRebind
    }

    override fun onRebind(intent: Intent) {
        // A client is binding to the service with bindService(),
        // after onUnbind() has already been called
    }
    private fun getLocation() {
        locationManager = getSystemService(Context.LOCATION_SERVICE) as LocationManager

        locationManager.requestLocationUpdates(LocationManager.GPS_PROVIDER, 5000, 5f, this)

    }
    override fun onLocationChanged(location: Location) {
        this.curLocation = location;
        curLatitude = this.curLocation!!.latitude;
        curLongitude = this.curLocation!!.longitude;
        saveSetting();
    }

    fun sendToDb(){
        Log.i("FirebaseDatabase", "Write!")
        val db = FirebaseFirestore.getInstance()
        val events = rawEvents
        rawEvents = ArrayList<PukEvent>()


        for (event  in events  ){
            // Create a new user with a first and last name
            val lat= event.latitude?:curLatitude
            val lon= event.longitude?:curLongitude
            val ev: HashMap<String, Any> = HashMap()

            event.time?.let {
                ev.put("time", it)
                ev.put("time_ms", it.time)
            }
            val now = Date()
            ev.put("score", event.score)
            lat?.let { ev.put("latitude", it) }
            lon?.let { ev.put("longitude", it) }
            event.userId?.let { ev.put("userId", it) }
            event.type?.let { ev.put("type", it) }
            ev.put("maxIndex", event.maxIndex)
            ev.put("maxValue", event.maxValue)
            ev.put("created", FieldValue.serverTimestamp())
            ev.put("createdLocal", now)
            ev.put("createdLocal_mc", now.time)
            // Add a new document with a generated ID

            db.collection("events")
                .add(ev)
                .addOnSuccessListener { documentReference ->
                    Log.d(TAG, "DocumentSnapshot added with ID: ${documentReference.id}")
                    pukEvents.add(event);
                }
                .addOnFailureListener { e ->
                    Log.w(TAG, "Error adding document", e)
                    rawEvents.add(event)
                }
        }
    }

}