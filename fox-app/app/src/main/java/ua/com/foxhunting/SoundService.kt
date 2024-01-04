package ua.com.foxhunting

import android.app.*
import android.app.NotificationChannel.DEFAULT_CHANNEL_ID
import android.content.Context
import android.content.Intent
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.media.AudioFormat.*
import android.media.AudioRecord
import android.os.*
import android.util.Log
import android.widget.Toast
import androidx.annotation.RequiresApi
import androidx.core.app.NotificationCompat
import androidx.core.app.NotificationManagerCompat
import com.google.firebase.firestore.FieldValue
import com.google.firebase.firestore.FirebaseFirestore
import org.tensorflow.lite.support.audio.TensorAudio
import org.tensorflow.lite.support.label.Category
import org.tensorflow.lite.task.audio.classifier.AudioClassifier
import java.io.File
import java.io.FileNotFoundException
import java.io.FileOutputStream
import java.io.IOException
import java.lang.System.`in`
import java.text.DateFormat
import java.text.SimpleDateFormat
import java.util.*
import kotlin.concurrent.scheduleAtFixedRate
import kotlin.experimental.and
import kotlin.concurrent.thread

class SoundService : Service(), LocationListener {

    private val FOREGROUND_ID: Int = 1024;
    private val RECORDER_BPP = 16
    private val AUDIO_RECORDER_FILE_EXT_WAV = ".wav"
    private val AUDIO_RECORDER_FOLDER = "Recorders"
    private val RECORDER_SAMPLERATE = 16000

    private var lastNotif: Date = Date()
    var PREF_NAME = "fox"
    var USER_ID_KEY = "user_id"
    var LATTITUDE_KEY = "latitude"
    var LONGITUDE_KEY = "longitude"

    var PROBABILITY_THRESHOLD_KEY = "probabilityThreshold"
    var PROBABILITY_THRESHOLD_OLD_KEY = "probabilityThresholdOld"

    var LogOut:String =""


    private var sampleRate: Int=0
    var TAG = "SoundService"

    // Binder given to clients
    private val binder = SoundBinder()

    private var allowRebind: Boolean = false
    public var isStarted: Boolean = true

    var modelPath = "pushki_model.tflite"
    var probabilityThreshold: Float = 0.80f
    var probabilityThresholdOld: Float = 0.80f
    var silence: String = "Silence"
    var pushka: String = "pushka"
    var timeout: Int = 4000
    lateinit var userId: String

    private  lateinit var tensorBuffers: TensorAudio;
    private lateinit var tensor: TensorAudio;
    private  var soundWindow:ArrayList<Float> = ArrayList<Float>()
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
    private var NOTIFICATION_CHANNEL_ID = "10201"
    private var notificationId = 10000
    private var lastEventTime:Date = Date()
    private lateinit var pendingIntent: PendingIntent

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
        editor.putFloat(PROBABILITY_THRESHOLD_KEY, probabilityThreshold)
        editor.putFloat(PROBABILITY_THRESHOLD_OLD_KEY, probabilityThresholdOld)
        editor.apply()
        editor.commit()

    }
    override fun onCreate() {
        // The service is being created

        pukEvents = ArrayList()
        rawEvents = ArrayList()
        pukNotofocation = ArrayList()
        var userSetting = getSharedPreferences(PREF_NAME,  MODE_PRIVATE);
        var defUserId ="User"+ (Date().time %1000).toString()
        userId = userSetting.getString(USER_ID_KEY, defUserId).toString()
        curLatitude = userSetting.getFloat(LATTITUDE_KEY, 0F).toDouble()
        curLongitude = userSetting.getFloat(LONGITUDE_KEY, 0F).toDouble()
        probabilityThreshold = userSetting.getFloat(PROBABILITY_THRESHOLD_KEY, probabilityThreshold)
        probabilityThresholdOld = userSetting.getFloat(PROBABILITY_THRESHOLD_OLD_KEY, probabilityThresholdOld)

        saveSetting();
        getLocation();
        try{
        // TODO 2.3: Loading the model from the assets folder
        classifier = AudioClassifier.createFromFile(this, modelPath)

        // TODO 3.1: Creating an audio recorder
        tensor = classifier.createInputTensorAudio()
            tensorBuffers = classifier.createInputTensorAudio()

        // TODO 3.2: showing the audio recorder specification
        val format = classifier.requiredTensorAudioFormat
        recorderSpecs = "Number Of Channels: ${format.channels}\n" +
                "Sample Rate: ${format.sampleRate}"
        sampleRate= format.sampleRate

        // TODO 3.3: Creating
        record = classifier.createAudioRecord()
        record.startRecording()
            createNotificationChannel()
    }catch (e: Exception) {
        Toast.makeText(
            applicationContext,
            "model. "+e.message,
            Toast.LENGTH_SHORT
        ).show()
        Log.i("ML", e.toString())
    }
        /*
        Timer().scheduleAtFixedRate(1, 500) {
            NextRawSound()
        }

         */

    }
    private fun runRawSound() {
        thread {
            while(isStarted) {
                Thread.sleep(500)
                nextRawSound()
            }
            // 알림 중단
            stopForeground(true)
            // 서비스 중단
            stopSelf()
            // 완료 알림 보내기
            // notificationManager.notify(NOTIFICATION_CHANNEL_ID, createCompleteNotification())
        }
    }

    private fun nextRawSound() {
        try {

            val numberOfSamples = tensor.load(record)
            soundWindow.addAll(tensor.tensorBuffer.floatArray.toList())
            if (soundWindow.size > sampleRate * 1.5) {
                val newArray = soundWindow.takeLast(sampleRate)
                soundWindow.clear();
                soundWindow.addAll(newArray)
            }
            tensor.load(soundWindow.toFloatArray())

            val output = classifier.classify(tensor)

            // TODO 4.2: Filtering out classifications with low probability
            val filteredModelOutputOld = output[0].categories.filter {
                it.score > probabilityThresholdOld
            }.sortedBy { -it.score }

            var modelOld: Category? = null;
            if (filteredModelOutputOld.isNotEmpty())
                modelOld = filteredModelOutputOld[0];

            val filteredModelOutput = output[1].categories.filter {
                it.score > probabilityThreshold
            }.sortedBy { -it.score }
            var model: Category? = null;
            if (filteredModelOutput.isNotEmpty())
                model = filteredModelOutput[0];

            // TODO 4.3: Creating a multiline string with the filtered results
            var outputStrOld =
                filteredModelOutputOld.sortedBy { -it.score }
                    .joinToString(separator = "\n") { "${it.label} ->  " + "%.2f".format(it.score) }
            var outputStr =
                filteredModelOutput.sortedBy { -it.score }
                    .joinToString(separator = "\n") { "${it.label} -> " + "%.2f".format(it.score) }

            // TODO 4.4: Updating the UI
            if (outputStr.isNotEmpty()) {
                LogOut = outputStr + "\n" + outputStrOld
                /*
                    runOnUiThread {
                        tvOutput.text = outputStr + outputStrOld
                    }
                    */

            }
            if (modelOld == null || modelOld.label != silence) {
                // Dont silence
                if (model != null && model.label == pushka) {
                    val currentTime = Date()
                    if ((currentTime.time - lastEventTime.time) > timeout) {
                        lastEventTime = currentTime
                        var event = PukEvent()
                        event.score = model.score;
                        if (curLatitude != null) {
                            event.latitude = curLatitude
                            event.longitude = curLongitude
                        }
                        event.type = model.label;
                        event.userId = userId;
                        var buffer = tensor.tensorBuffer.floatArray
                        var maxIndex: Int = 0
                        var maxValue: Float = 0F
                        for ((index, value) in buffer.withIndex()) {
                            if (Math.abs(value) > maxValue) {
                                maxValue = Math.abs(value)
                                maxIndex = index
                            }
                        }
                        event.maxIndex = maxIndex;
                        event.maxValue = maxValue;
                        // Add offset
                        var offSet = (buffer.size - 1 - maxIndex) * 1000 / sampleRate
                        currentTime.time = (currentTime.time - offSet).toLong()
                        event.time = currentTime


                        rawEvents.add(event)
                        sendToDb()
                        val df: DateFormat = SimpleDateFormat("HH:mm:ss")
                        val msg = df.format(event.time) + " " + "%.3f".format(event.maxValue)
                        pukNotofocation.add(msg)
                        notifications()
                        var intBuffer = ShortArray(buffer.size)
                        for ((index, value) in buffer.withIndex()) {
                            intBuffer[index] = (value * Short.MAX_VALUE).toInt().toShort()
                        }
                        var path: String = getFilename(msg)
                        writeWaveFile(intBuffer, path)
                    }
                }
            }
        } catch (e: Exception) {
            Log.i("Yamnet", e.toString());
        }
    }

    @RequiresApi(Build.VERSION_CODES.O)
    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        // If the notification supports a direct reply action, use
// PendingIntent.FLAG_MUTABLE instead.
        pendingIntent =
            Intent(this, MainActivity::class.java).let { notificationIntent ->
                PendingIntent.getActivity(this, 0, notificationIntent,
                    PendingIntent.FLAG_IMMUTABLE)
            }

        val notification: Notification = Notification.Builder(this, NOTIFICATION_CHANNEL_ID)
            .setContentTitle(getText(R.string.notification_title))
            .setContentText(getText(R.string.notification_message))
            .setSmallIcon(R.drawable.notification_icon)
            .setContentIntent(pendingIntent)
            .setTicker(getText(R.string.ticker_text))
            .setAutoCancel(true)
            .setTimeoutAfter(2000)
            .build()

// Notification ID cannot be 0.
        startForeground(FOREGROUND_ID, notification)
        runRawSound();
        return Service.START_STICKY;
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
        try{
            locationManager = getSystemService(Context.LOCATION_SERVICE) as LocationManager
            // LocationManager.GPS_PROVIDER
            locationManager.requestLocationUpdates(LocationManager.NETWORK_PROVIDER, 5000, 5f, this)
        }catch (e: Exception) {
            Toast.makeText(
                applicationContext,
                "location:  "+e.message,
                Toast.LENGTH_SHORT
            ).show()
            Log.i("LCOATION", e.toString())
        }

    }
    override fun onStatusChanged(provider: String, status: Int, extras: Bundle) {}
    override fun onProviderEnabled(provider: String) {}
    override fun onProviderDisabled(provider: String) {}
    override fun onLocationChanged(location: Location) {
        this.curLocation = location;
        curLatitude = this.curLocation!!.latitude;
        curLongitude = this.curLocation!!.longitude;
        saveSetting();
    }
    private fun createNotificationChannel() {
        // Create the NotificationChannel, but only on API 26+ because
        // the NotificationChannel class is new and not in the support library
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val name = getString(R.string.channel_name)
            val descriptionText = getString(R.string.channel_description)
            val importance = NotificationManager.IMPORTANCE_DEFAULT
            val channel = NotificationChannel(NOTIFICATION_CHANNEL_ID, name, importance).apply {
                description = descriptionText
            }
            // Register the channel with the system
            val notificationManager: NotificationManager =
                getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            notificationManager.createNotificationChannel(channel)
        }else{
            NOTIFICATION_CHANNEL_ID = "";
        }
    }
    fun notifications(){
        if(pukNotofocation.size>0){
            val textContent=pukNotofocation.joinToString(separator = "\n") { it }
            var  textTitle:String
            if(pukNotofocation.size>1)
                textTitle ="${pukNotofocation.size} FH events"
            else
                textTitle ="FH events"
            var builder = NotificationCompat.Builder(this, NOTIFICATION_CHANNEL_ID)
                .setSmallIcon(R.drawable.notification_icon)
                .setContentTitle(textTitle)
                .setContentText(textContent)
                .setContentIntent(pendingIntent)
                .setPriority(NotificationCompat.PRIORITY_DEFAULT)
                .setAutoCancel(true)
                .setTimeoutAfter(20000)
            with(NotificationManagerCompat.from(this)) {
                // notificationId is a unique int for each notification that you must define
                notify(notificationId, builder.build())
            }
        }
        if(Date().time - lastNotif.time>10000 ) {
            notificationId +=1;
            lastNotif = Date()
            pukNotofocation.clear()
        }


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
    @Throws(IOException::class)
    private fun WriteWaveFileHeader(
        out: FileOutputStream, totalAudioLen: Long,
        totalDataLen: Long, longSampleRate: Long, channels: Int,
        byteRate: Long
    ) {
        val header = ByteArray(44)
        header[0] = 'R'.code.toByte() // RIFF/WAVE header
        header[1] = 'I'.code.toByte()
        header[2] = 'F'.code.toByte()
        header[3] = 'F'.code.toByte()
        header[4] = (totalDataLen and 0xff).toByte()
        header[5] = (totalDataLen shr 8 and 0xff).toByte()
        header[6] = (totalDataLen shr 16 and 0xff).toByte()
        header[7] = (totalDataLen shr 24 and 0xff).toByte()
        header[8] = 'W'.code.toByte()
        header[9] = 'A'.code.toByte()
        header[10] = 'V'.code.toByte()
        header[11] = 'E'.code.toByte()
        header[12] = 'f'.code.toByte() // 'fmt ' chunk
        header[13] = 'm'.code.toByte()
        header[14] = 't'.code.toByte()
        header[15] = ' '.code.toByte()
        header[16] = 16 // 4 bytes: size of 'fmt ' chunk
        header[17] = 0
        header[18] = 0
        header[19] = 0
        header[20] = 1 // format = 1
        header[21] = 0
        header[22] = channels.toByte()
        header[23] = 0
        header[24] = (longSampleRate and 0xff).toByte()
        header[25] = (longSampleRate shr 8 and 0xff).toByte()
        header[26] = (longSampleRate shr 16 and 0xff).toByte()
        header[27] = (longSampleRate shr 24 and 0xff).toByte()
        header[28] = (byteRate and 0xff).toByte()
        header[29] = (byteRate shr 8 and 0xff).toByte()
        header[30] = (byteRate shr 16 and 0xff).toByte()
        header[31] = (byteRate shr 24 and 0xff).toByte()
        header[32] = (1 * 16 / 8).toByte() // block align
        header[33] = 0
        header[34] = RECORDER_BPP.toByte() // bits per sample
        header[35] = 0
        header[36] = 'd'.code.toByte()
        header[37] = 'a'.code.toByte()
        header[38] = 't'.code.toByte()
        header[39] = 'a'.code.toByte()
        header[40] = (totalAudioLen and 0xff).toByte()
        header[41] = (totalAudioLen shr 8 and 0xff).toByte()
        header[42] = (totalAudioLen shr 16 and 0xff).toByte()
        header[43] = (totalAudioLen shr 24 and 0xff).toByte()
        out.write(header, 0, 44)
    }
    private fun getFilename(name:String): String {
        var nameFile = name.replace(" ","_").replace(",","_").replace(":","_")
        // var filepath = this.getFilesDir()
        var filepath = getExternalFilesDir(null)
        val file = File(filepath, AUDIO_RECORDER_FOLDER)
        if (!file.exists()) {
            var res = file.mkdirs()
        }
         return file.getAbsolutePath().toString() + "/" + nameFile +
                AUDIO_RECORDER_FILE_EXT_WAV
    }
    private fun writeWaveFile(bufferData:ShortArray, outFilename: String) {
        var out: FileOutputStream? = null
        var totalAudioLen: Long = 0
        var totalDataLen = totalAudioLen + 36
        val longSampleRate = sampleRate.toLong()
        val channels = 1
        val byteRate = (RECORDER_BPP * RECORDER_SAMPLERATE * channels / 8).toLong()
        val dataBytes = ByteArray(2)
        try {
            val f = File(outFilename)
            f.createNewFile()
            out = FileOutputStream(f)
            totalAudioLen = (bufferData.size*2).toLong()
            totalDataLen = totalAudioLen + 36
            Log.i("wav", "File size: $totalDataLen")
            WriteWaveFileHeader(
                out, totalAudioLen, totalDataLen,
                longSampleRate, channels, byteRate
            )
                for ( v in bufferData){

                    dataBytes[0] = (v.toInt()  and 0xff).toByte()
                    dataBytes[1] = (v.toInt() shr 8 and 0xff).toByte()
                    out.write(dataBytes)
                }
                out.close()

        } catch (e: FileNotFoundException) {
            e.printStackTrace()
        } catch (e: IOException) {
            e.printStackTrace()
        }
    }
}