//https://code.stypi.com/rycxdugo

// params
// set these to your system camera
int camera_i = 3;
int camera_h = 640;
int camera_w = 480;
int serial_port = 6;
Boolean debug = true;
Boolean print_stuff = false;

//GUI
int screen_width = 690;
int screen_height = 700;
int w_ctr = screen_width/2;
int h_ctr = screen_height/2;
int spacing = 25;

//timing
int system_time             = 0; //the system time (called each time in the loop)
int recording_time_start    = 0; // recording_time_start called on pressing 'r'
int recording_time_current  = 0; 
int recording_time_last     = 0; // time diff between this image and the previous one
int recording_time_elapsed  = 0;
int recording_frame_interval = 10;

//webcam
import processing.video.*;
Capture webcam;
Boolean record_video = false;
String frame_filename;
String desktop_path, data_path;

// arduino

import processing.serial.*;  // serial library lets us talk to Arduino
PFont font;
Serial port;
float sensor_value = 0.0;


void setup(){
	size(screen_width, screen_height);
	frameRate(2000);
	background(255);
	desktop_path = System.getProperty("user.home")+"/Desktop/Massive_BIG_Data/collected_data/";
	connect_webcam();
	connect_serial();
}

void draw(){}

void keyPressed(){
	if (key=='r'){
		record_video = !record_video;
		recording_time_start = millis();
		recording_time_current  = 0;
		recording_time_elapsed  = 0;
		recording_time_last     = 0;
		if(print_stuff) println("recording = " + record_video.toString() + " " + recording_time_start);
	}
}

//-------------------------------------------------------------------------------------------------------------
//Webcam

void connect_webcam(){
	try{
	    String[] camera_names = Capture.list();
	    if(print_stuff) println(camera_names);

	    // SUPER IMPORTANT!!! the camera code may only work in one of the two below configurations
	    webcam = new Capture(this, camera_names[camera_i]);
	    webcam.start();

	    alert_window("Webcam detected. Using default device.");
	} catch(Exception e) {
	    alert_window("Aw nah! Your webcam is missing. Please attach a webcam and restart the program.");
	}
}

/*
	webcam = new Capture(this, 640, 360, camera_names[3], 30);
	webcam = new Capture(this, camera_h, camera_w, 30);
	webcam.settings();
*/

void captureEvent(Capture webcam) {
	if(record_video) {
		system_time = millis();
		recording_time_current = system_time - recording_time_start;
		recording_time_elapsed = recording_time_current - recording_time_last;
		
		if (recording_time_elapsed > recording_frame_interval){
			try{
				if(webcam.available()){
					// read
					webcam.read();
					PImage webcam_image = webcam.get(0, 0, camera_h, camera_w);
					smooth(); image(webcam_image, spacing, 150 + spacing, camera_h, camera_w); noSmooth();
					
					// update time before saving image to disk so captureEvent can run again without waiting
					recording_time_last = recording_time_current;
					
					// write
					frame_filename = data_path + "images/" + "ms_" + nf(recording_time_current, 10) + "_sensor_" + sensor_value + "_.bmp";
					if(!debug){ webcam_image.save(frame_filename); }
					
					// display
					alert_window("recording " + recording_time_current + " ms\n" + "sensor " + sensor_value);
				}
			} catch(Exception e) {
				if(print_stuff) println("capture error!!");
			}
		}
	} else {
		if(print_stuff) println("not recording");
	}
}

//-------------------------------------------------------------------------------------------------------------
// serial

void connect_serial(){
    println(Serial.list());                             // print a list of available serial ports
    port = new Serial(this, Serial.list()[serial_port], 115200);  // choose the right port and set the baud rate
    port.bufferUntil('\n');                             // arduino will end each data packet with a carriage return 
    port.clear();                                       // flush the Serial buffer, it's very hygienic
}

void serialEvent(Serial port){   
   String inData = port.readStringUntil('\n');  // read the ascii data into a String

  if (inData.charAt(0) == 'S'){           // leading 'S' means Pulse Sensor data packet
     inData = inData.substring(1);        // cut off the leading 'S'
     sensor_value = float(trim(inData));         // trim the \n off the end
     //int newPPG = int(inData);          // convert ascii string to integer
   }   
}

//-------------------------------------------------------------------------------------------------------------
// general functions

void alert_window(String message_text){
	int padding = 20;
	int pos_width = 100;
	int pos_height = 100;

	//rectangle
	noStroke();
	rectMode(CORNERS);
	fill(240);
	rect(spacing, spacing, camera_h + spacing, 150);

	//text
	fill(0);
	textAlign(LEFT);

	text(message_text, spacing+padding, spacing+padding, camera_h-2*padding, 150-padding);
	if(print_stuff) println(message_text);
}

void draw_grid(int pix, color stroke_color, int stroke_weight){
	stroke(stroke_color);
	strokeWeight(stroke_weight);
	for(int i=0; i<width;  i=i+pix){ line(i, 0, i, height); }
	for(int i=0; i<height; i=i+pix){ line(0, i, width, i); }
	noStroke();
}


