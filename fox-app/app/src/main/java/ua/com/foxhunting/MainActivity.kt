// Copyright 2022 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package ua.com.foxhunting

//import com.google.android.gms.auth.api.signin.GoogleSignIn
//import com.google.android.gms.auth.api.signin.GoogleSignInClient
// import com.google.firebase.auth.FirebaseAuth
//import com.google.firebase.auth.ktx.auth
// import com.google.firebase.firestore.ktx.firestore
// import com.google.firebase.database.ServerValue
// import com.google.firebase.ktx.Firebase
import android.Manifest
import android.accounts.Account
import android.accounts.AccountManager
import android.content.Context
import android.content.pm.PackageManager
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.media.AudioRecord
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.EditText
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.widget.addTextChangedListener
import com.google.android.things.device.TimeManager
import com.google.firebase.firestore.FieldValue
import com.google.firebase.firestore.FirebaseFirestore
import org.tensorflow.lite.support.audio.TensorAudio
import org.tensorflow.lite.support.label.Category
import org.tensorflow.lite.task.audio.classifier.AudioClassifier
import java.lang.Math.abs
import java.text.DateFormat
import java.text.SimpleDateFormat
import java.util.*
import kotlin.concurrent.scheduleAtFixedRate


class MainActivity : AppCompatActivity(), LocationListener {
    private var sampleRate: Int=0
    var TAG = "MainActivity"
    var USER_ID_KEY = "user_id"
    var LATTITUDE_KEY = "latitude"
    var LONGITUDE_KEY = "longitude"


    // TODO 2.1: defines the model to be used
    var modelPath = "pushki_model.tflite"

    // TODO 2.2: defining the minimum threshold
    var probabilityThreshold: Float = 0.75f
    var probabilityThresholdOld: Float = 0.8f
    var silence: String = "Silence"
    var pushka: String = "pushka"
    lateinit var userId: String
    lateinit var tvOutput: TextView
    private lateinit var locationManager: LocationManager
    private val locationPermissionCode = 2
    private lateinit var tvGpsLocation: TextView
    private lateinit var tvLog: TextView
    private lateinit var editUserId: EditText

    private lateinit var tensor: TensorAudio;
    private lateinit var record: AudioRecord;
    private lateinit var classifier: AudioClassifier;
    private lateinit var pukEvents: ArrayList<PukEvent>;

    lateinit var timeManager: TimeManager;
    lateinit var calendar: Calendar;
    var curLocation:Location?=null;
    var  curLatitude:Double?=null
    var  curLongitude:Double?=null
    //private lateinit var auth: FirebaseAuth
    //private lateinit var googleSignInClient: GoogleSignInClient
    private val REQ_ONE_TAP = 2  // Can be any integer unique to the Activity
    private var showOneTapUI = true

    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)
        outState.putString(USER_ID_KEY, userId)
        if(curLatitude!=null)
            outState.putDouble(LATTITUDE_KEY, curLatitude!!)
        if(curLongitude!=null)
        outState.putDouble(LONGITUDE_KEY, curLongitude!!)
    }
    override fun onStart() {
        super.onStart()
        // Check if user is signed in (non-null) and update UI accordingly.
        //val currentUser = auth.currentUser
        //updateUI(currentUser)

    }
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // [START config_signin]
        // Configure Google Sign In
        /*
        val gso = GoogleSignInOptions.Builder(GoogleSignInOptions.DEFAULT_SIGN_IN)
            .requestIdToken(getString(R.string.default_web_client_id))
            .requestEmail()
            .build()

        googleSignInClient = GoogleSignIn.getClient(this, gso)
        */

        // [END config_signin]


        // [START initialize_auth]
        // Initialize Firebase Auth
        // auth = Firebase.auth
        // [END initialize_auth]
        editUserId.addTextChangedListener { userId = it.toString() }

        pukEvents = ArrayList()
        userId = ""
        if (savedInstanceState != null) {
            val u  = savedInstanceState.getString(USER_ID_KEY)
            if(u!=null) userId = u
            if(savedInstanceState.containsKey(LATTITUDE_KEY)) {
                curLatitude  = savedInstanceState.getDouble(LATTITUDE_KEY)
                curLongitude = savedInstanceState.getDouble(LONGITUDE_KEY)
            }
        }
        if(userId=="") {
            userId = UUID.randomUUID().toString()
            val am: AccountManager = AccountManager.get(this) // "this" references the current Context
            val accounts: Array<out Account> = am.getAccountsByType("com.google")
            if(accounts.size>0){
                userId = accounts.get(0).name
            }
        }

        getLocation();

        try {
            calendar = Calendar.getInstance();
            timeManager = TimeManager.getInstance()
            timeManager.setAutoTimeEnabled(true)

        }catch (e: Exception) {
            val toast = Toast.makeText(
                applicationContext,
                "timeManager. "+e.message,
                Toast.LENGTH_LONG
            )
            Log.i("Yamnet", e.toString());
            toast.show()
        }

        try {
        val REQUEST_RECORD_AUDIO = 1337
        requestPermissions(arrayOf(Manifest.permission.RECORD_AUDIO), REQUEST_RECORD_AUDIO)

        tvOutput = findViewById<TextView>(R.id.output)
            tvLog = findViewById<TextView>(R.id.tvLog)
        val recorderSpecsTextView = findViewById<TextView>(R.id.textViewAudioRecorderSpecs)

        // TODO 2.3: Loading the model from the assets folder
        classifier = AudioClassifier.createFromFile(this, modelPath)

        // TODO 3.1: Creating an audio recorder
        tensor = classifier.createInputTensorAudio()

        // TODO 3.2: showing the audio recorder specification
        val format = classifier.requiredTensorAudioFormat
        val recorderSpecs = "Number Of Channels: ${format.channels}\n" +
                "Sample Rate: ${format.sampleRate}"
            sampleRate= format.sampleRate
        recorderSpecsTextView.text = recorderSpecs

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
        Log.i("Yamnet", "scheduleAtFixedRate!")
    Timer().scheduleAtFixedRate(1, 500) {
            try{
            // TODO 4.1: Classifing audio data

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
            if (outputStr.isNotEmpty())
                runOnUiThread {
                        tvOutput.text = outputStr+outputStrOld
                }
                if(modelOld==null || modelOld.label!=silence){
                    // Dont silence
                    if(model!=null && model.label==pushka){
                        val  currentTime = calendar.getTime();
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
                                if( abs(value)>maxValue){
                                    maxValue = abs(value)
                                    maxIndex = index
                                }
                        }
                        event.maxIndex = maxIndex;
                        event.maxValue = maxValue;
                        // Add offset
                        var offSet = (buffer.size-1-maxValue)*1000/sampleRate
                        currentTime.time = (currentTime.time-offSet).toLong()
                        event.time =currentTime


                            pukEvents.add(event)
                        runOnUiThread {
                            val df: DateFormat = SimpleDateFormat("HH:mm:ss")
                            tvLog.text = df.format(event.time)+ " "+"%.3f".format(event.maxValue) + " "+" - ${model.label}\n" + tvLog.text
                        }
                        sendToDb()
                    }
                }
            }catch (e: Exception) {
                Log.i("Yamnet", e.toString());
            }
        }
    }
    /*
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        // Result returned from launching the Intent from GoogleSignInApi.getSignInIntent(...);
        if (requestCode == RC_SIGN_IN) {
            val task = GoogleSignIn.getSignedInAccountFromIntent(data)
            try {
                // Google Sign In was successful, authenticate with Firebase
                val account = task.getResult(ApiException::class.java)!!
                Log.d(TAG, "firebaseAuthWithGoogle:" + account.id)
                firebaseAuthWithGoogle(account.idToken!!)
            } catch (e: ApiException) {
                // Google Sign In failed, update UI appropriately
                Log.w(TAG, "Google sign in failed", e)
            }
        }
    }
    private fun updateUI(user: FirebaseUser?) {

    }
    */

    private fun getLocation() {
        locationManager = getSystemService(Context.LOCATION_SERVICE) as LocationManager
        if ((ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED)) {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.ACCESS_FINE_LOCATION), locationPermissionCode)
        }
        locationManager.requestLocationUpdates(LocationManager.GPS_PROVIDER, 5000, 5f, this)

    }
    override fun onLocationChanged(location: Location) {
        this.curLocation = location;
        curLatitude = this.curLocation!!.latitude;
        curLongitude = this.curLocation!!.longitude;

        tvGpsLocation = findViewById(R.id.tvGpsLocation)
        tvGpsLocation.text = "Latitude: " + "%.3f".format(location.latitude) + " , Longitude: " + "%.3f".format(location.longitude)
    }
    override fun onRequestPermissionsResult(requestCode: Int,
                                            permissions: Array<out String>,
                                            grantResults: IntArray) {
        if (requestCode == locationPermissionCode) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                Toast.makeText(this, "Permission Granted", Toast.LENGTH_SHORT).show()
            }
            else {
                Toast.makeText(this, "Permission Denied", Toast.LENGTH_SHORT).show()
            }
        }
    }
    fun sendToDb(){
        Log.i("FirebaseDatabase", "Write!")
        val db = FirebaseFirestore.getInstance()



        for (event  in pukEvents){
            // Create a new user with a first and last name
            val lat= event.latitude?:curLatitude
            val lon= event.longitude?:curLongitude
            val ev: HashMap<String, Any> = HashMap()

            event.time?.let {
                ev.put("time", it)
                ev.put("time_ms", it.time)
            }
            ev.put("score", event.score)
            lat?.let { ev.put("latitude", it) }
            lon?.let { ev.put("longitude", it) }
            event.userId?.let { ev.put("userId", it) }
            event.type?.let { ev.put("type", it) }
            ev.put("maxIndex", event.maxIndex)
            ev.put("maxValue", event.maxValue)
             ev.put("created", FieldValue.serverTimestamp())
            ev.put("createdLocal", calendar.getTime())
            ev.put("createdLocal_mc", calendar.getTime().time)
            // Add a new document with a generated ID

            db.collection("events")
                .add(ev)
                .addOnSuccessListener { documentReference ->
                    Log.d(TAG, "DocumentSnapshot added with ID: ${documentReference.id}")
                }
                .addOnFailureListener { e ->
                    Log.w(TAG, "Error adding document", e)
                }
        }


        pukEvents.clear()
    }
}