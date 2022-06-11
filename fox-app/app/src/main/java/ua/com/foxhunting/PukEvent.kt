package ua.com.foxhunting

import android.location.Location
import java.util.*

class PukEvent {
   var score: Float=0F
   var longitude: Double?=null
   var latitude: Double?=null
   var maxValue: Float=0F
   var maxIndex: Int=0
   var time: Date?=null
   var loc:Location?=null
   var type: String?=null
   var userId: String?=null
}
