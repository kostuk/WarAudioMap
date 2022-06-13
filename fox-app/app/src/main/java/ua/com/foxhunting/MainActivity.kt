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
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.content.pm.PackageManager
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.media.AudioRecord
import android.os.Bundle
import android.os.IBinder
import android.util.Log
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.widget.addTextChangedListener
import com.google.android.gms.auth.api.identity.BeginSignInRequest
import com.google.android.gms.auth.api.identity.Identity
import com.google.android.gms.auth.api.identity.SignInClient
import com.google.android.gms.auth.api.signin.GoogleSignIn
import com.google.android.gms.common.api.ApiException
import com.google.android.things.device.TimeManager
import com.google.firebase.auth.FirebaseAuth
import com.google.firebase.auth.FirebaseUser
import com.google.firebase.auth.GoogleAuthProvider
import com.google.firebase.auth.ktx.auth
import com.google.firebase.firestore.FieldValue
import com.google.firebase.firestore.FirebaseFirestore
import com.google.firebase.ktx.Firebase
import org.tensorflow.lite.support.audio.TensorAudio
import org.tensorflow.lite.support.label.Category
import org.tensorflow.lite.task.audio.classifier.AudioClassifier
import java.lang.Math.abs
import java.text.DateFormat
import java.text.SimpleDateFormat
import java.util.*
import kotlin.concurrent.scheduleAtFixedRate


class MainActivity : AppCompatActivity() {
    private lateinit var  oneTapClient: SignInClient
    private var sampleRate: Int=0
    var TAG = "MainActivity"


    lateinit var tvOutput: TextView
    private lateinit var locationManager: LocationManager
    private val locationPermissionCode = 2
    private lateinit var tvGpsLocation: TextView
    private lateinit var tvLog: TextView
    private lateinit var tvAudioRecorderSpecs: TextView

    private lateinit var editUserId: EditText

    private lateinit var tensor: TensorAudio;
    private lateinit var record: AudioRecord;
    private lateinit var classifier: AudioClassifier;
    private lateinit var pukEvents: ArrayList<PukEvent>;
    private lateinit var mService: SoundService
    private var mBound: Boolean = false

    lateinit var timeManager: TimeManager;
    lateinit var calendar: Calendar;
    var curLocation:Location?=null;
    var  curLatitude:Double?=null
    var  curLongitude:Double?=null
    private lateinit var auth: FirebaseAuth
    private val REQ_ONE_TAP = 2  // Can be any integer unique to the Activity
    private var showOneTapUI = true

    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)

    }
    /* Defines callbacks for service binding, passed to bindService()  */
    private val connection = object : ServiceConnection {

        override fun onServiceConnected(className: ComponentName, service: IBinder) {
            // We've bound to LocalService, cast the IBinder and get LocalService instance
            val binder = service as SoundService.SoundBinder
            mService = binder.getService()
            mBound = true
            editUserId.setText(mService.userId)
            tvAudioRecorderSpecs.text = mService.recorderSpecs

        }

        override fun onServiceDisconnected(arg0: ComponentName) {
            mBound = false
        }
    }
    override fun onStart() {
        super.onStart()
        // Check if user is signed in (non-null) and update UI accordingly.
        var currentUser = auth.currentUser
        updateUI(currentUser)
        Intent(this, SoundService::class.java).also { intent ->
            bindService(intent, connection, Context.BIND_AUTO_CREATE)
        }

    }
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        // [START config_signin]
        // Configure Google Sign In
        oneTapClient = Identity.getSignInClient(this)
        var signInRequest = BeginSignInRequest.builder()
            .setGoogleIdTokenRequestOptions(
                BeginSignInRequest.GoogleIdTokenRequestOptions.builder()
                .setSupported(true)
                // Your server's client ID, not your Android client ID.
                .setServerClientId(getString(R.string.default_web_client_id))
                // Only show accounts previously used to sign in.
                .setFilterByAuthorizedAccounts(true)
                .build())
        oneTapClient.beginSignIn(signInRequest.build())

        // [END config_signin]


        // [START initialize_auth]
        // Initialize Firebase Auth
        auth = Firebase.auth
        // [END initialize_auth]


       /*
        if(userId=="") {
            userId = UUID.randomUUID().toString()
            val am: AccountManager = AccountManager.get(this) // "this" references the current Context
            val accounts: Array<out Account> = am.getAccountsByType("com.google")
            if(accounts.size>0){
                userId = accounts.get(0).name
            }
        }

        */
        editUserId = findViewById<EditText>(R.id.editUserId)
        tvLog = findViewById<EditText>(R.id.tvLog)
        tvGpsLocation = findViewById<EditText>(R.id.tvGpsLocation)
        tvOutput = findViewById<EditText>(R.id.output)
        tvAudioRecorderSpecs = findViewById<EditText>(R.id.tvAudioRecorderSpecs)

        findViewById<Button>(R.id.btSetUserId).setOnClickListener {
            if(mBound){
                mService.userId = editUserId.text.toString()
                mService.recorderSpecs
                mService.saveSetting()
            }
        }
        /*
        editUserId.addTextChangedListener { userId = it.toString() }
        editUserId.setText(userId)
        */
        getLocation();
        try {
            calendar = Calendar.getInstance();
            //timeManager = TimeManager.getInstance()
            //timeManager.setAutoTimeEnabled(true)

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
        requestPermissions(arrayOf(
            Manifest.permission.RECORD_AUDIO,
            Manifest.permission.ACCESS_FINE_LOCATION,
            Manifest.permission.GET_ACCOUNTS,
            Manifest.permission.INTERNET,
        ), REQUEST_RECORD_AUDIO)

    }catch (e: Exception) {
        Toast.makeText(
            applicationContext,
            "model. "+e.message,
            Toast.LENGTH_SHORT
        ).show()
            Log.i("ML", e.toString())
    }
        Log.i("Yamnet", "scheduleAtFixedRate!")
        Timer().scheduleAtFixedRate(1, 2000) {
            if(mBound){
                runOnUiThread {
                    tvOutput.text =  mService.LogOut
                    if(mService.curLatitude!=null) {
                        tvGpsLocation.text =
                            "Latitude: " + "%.3f".format(mService.curLatitude) + " , Longitude: " + "%.3f".format(
                                mService.curLongitude
                            )
                    }else{
                        tvGpsLocation.text="gsp ..."
                    }
                    var events:String=""
                    for (event in mService.pukEvents.reversed()){
                        val df: DateFormat = SimpleDateFormat("HH:mm:ss")
                        events += df.format(event.time)+ " "+"%.3f".format(event.maxValue) + " "+" - ${event.type}\n"
                    }
                    tvLog.text = events;
                    if(mService.pukNotofocation.size>0){
                        val m=mService.pukNotofocation.joinToString(separator = "\n") { it }
                        Toast.makeText(
                            applicationContext,
                            m,
                            Toast.LENGTH_LONG
                        ).show()
                        mService.pukNotofocation.clear()
                    }
                }
            }
        }
    }
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        val googleCredential = oneTapClient.getSignInCredentialFromIntent(data)
        val idToken = googleCredential.googleIdToken
            /*
        when {
            idToken != null -> {
                // Got an ID token from Google. Use it to authenticate
                // with Firebase.
                val firebaseCredential = GoogleAuthProvider.getCredential(idToken, null)
                auth.signInWithCredential(firebaseCredential)
                    .addOnCompleteListener(this) { task ->
                        if (task.isSuccessful) {
                            // Sign in success, update UI with the signed-in user's information
                            Log.d(TAG, "signInWithCredential:success")
                            val user = auth.currentUser
                            updateUI(user)
                        } else {
                            // If sign in fails, display a message to the user.
                            Log.w(TAG, "signInWithCredential:failure", task.exception)
                            updateUI(null)
                        }
                    }
            }
            else -> {
                // Shouldn't happen.
                Log.d(TAG, "No ID token!")
            }
        }

             */
    }
    private fun updateUI(user: FirebaseUser?) {

    }
    public  fun signOut(){
        Firebase.auth.signOut()
    }

    private fun getLocation() {
        locationManager = getSystemService(Context.LOCATION_SERVICE) as LocationManager
        if ((ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED)) {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.ACCESS_FINE_LOCATION), locationPermissionCode)
        }

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
}