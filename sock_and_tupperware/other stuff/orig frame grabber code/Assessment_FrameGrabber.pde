import processing.video.*;
import processing.net.*; 
import processing.serial.*;
//import cc.arduino.*;

//Objects
//Arduino arduino;
Client server_client;
Capture webcam;

//data output vars
String desktop_path, data_path;

//GUI
int screen_width = 690;
int screen_height = 700;
int w_ctr = screen_width/2;
int h_ctr = screen_height/2;
int spacing = 25;

//Server
String server_data;
Boolean server_connected = false;
int port = 5204;

//Assessment Vars
String trial_vars;
String subject_id;
Boolean record_video = false;
int channel_trigger = 0;
int channel_trigger_last = 0;

//timing
int trial_time_start;
int time_trial_current;
int trial_last = 0;
int trial = 0;
String data_last ="";
PImage sammich;

//Webcam
int capture_millis_last = 0;
int frame_interval = 2; //ms between frames

void setup()
{
	size(screen_width, screen_height);
	frameRate(2000);
	background(255);
	//setUpArduino();
	desktop_path = System.getProperty("user.home")+"/Desktop/Assessment Battery/collected_data/";
	//make data file
	// data_path = desktop_path + "/" +
	// 			"Assessment Battery/collected_data/" +
	// 			subject_id + "/" +
	// 			runsheet_name + "/";
	sammich = loadImage("sammich.jpg");
	//draw_grid(25, 0, 1);
}

void draw()
{
	if(!server_connected)
	{
		alert_window("Trying to connect to Assessment Software\nPinging localhost port " + port + " sent at " + millis() + " ms");
		image(sammich, spacing, 150 + spacing, 640, 480);
		connect_server();
		delay(1000);
	}
}

void keyPressed()
{
	if (key=='p') record_video = false;
}

//Connections
void connect_server()
{
	try
	{
		server_client = new Client(this, "127.0.0.1", port);
	} catch(Exception e)
	{
		alert_window("No server response. " + millis() );
	}
}

void clientEvent(Client server_client)
{
	if (server_client.available() > 0)
	{
		try
		{
			if(!server_connected) { server_connected = true;  connect_webcam(); } //frame_filename
			parse_server_data(server_client.readString());
			server_client.clear();
		}catch(Exception e) 
		{
			println("caught client event error");
		}
	}
}

void parse_server_data(String server_data)
{
	String[] server_data_arr = split(server_data, ",");

	if (server_data_arr.length > 1) 
	{		
		subject_id = server_data_arr[0];
		if(subject_id.equals("exit") ) { record_video = false; trial = -1; return; }
		trial_vars = server_data_arr[1];
		trial = int(server_data_arr[2]);
		record_video = server_data_arr[3].equals("true");
		channel_trigger = int(server_data_arr[4]);
		data_path = desktop_path + subject_id + "/DERT/";
		
		//if(channel_trigger != channel_trigger_last) trigger_arduino(channel_trigger);
		if(trial != trial_last) new_trial();
		//alert_window(trial_vars + "\nrecord: " + record_video.toString() + "\ntrial: " + trial + " channel " + channel_trigger);
	}
}

void new_trial()
{
	println("fg raw: " + server_data);
	//if(channel_trigger != channel_trigger_last) trigger_arduino(channel_trigger);
	println("new trial " + trial);
	trial_time_start = millis();
	time_trial_current = 0;
	trial_last = trial;
	channel_trigger_last = channel_trigger;
	//"_ms_" + time_trial_current + ".png"
}


//Ardiuno
void setUpArduino()
{
	//arduino = new Arduino(this, Arduino.list()[0], 57600);
	//arduino.pinMode(3, Arduino.OUTPUT);
	//arduino.pinMode(4, Arduino.OUTPUT);
}

void set_arduino_low()
{
	//arduino.digitalWrite(3, Arduino.LOW);
	//arduino.digitalWrite(4, Arduino.LOW);
}

void trigger_arduino(int channel)
{
	//if(channel == 3 || channel == 4) {arduino.digitalWrite(channel, Arduino.HIGH);} else { set_arduino_low(); }
}


//GUI
void alert_window(String message_text)
{
	int padding = 20;
	int pos_width = 100;
	int pos_height = 100;

	//rectangle
	noStroke();
	rectMode(CORNERS);
	fill(240);
	rect(spacing, spacing, 640 + spacing, 150);

	//text
	fill(0);
	textAlign(LEFT);

	text(message_text, spacing+padding, spacing+padding, 640-2*padding, 150-padding);
}

void draw_grid(int pix, color stroke_color, int stroke_weight)
{
	stroke(stroke_color);
	strokeWeight(stroke_weight);
	
	for(int i=0; i<width; i=i+pix)
	{
		line(i, 0, i, height);
	}
	
	for(int i=0; i<height; i=i+pix)
	{
		line(0, i, width, i);
	}
	
	noStroke();
}


//Webcam
void connect_webcam()
{
	try
	{
		String[] camera_names = Capture.list();
		println(camera_names);

		//webcam = new Capture(this, 640, 480, camera_names[1], 30);
		webcam = new Capture(this, 640, 480, 30);
		//webcam.settings();
		alert_window("Webcam detected. Using default device.");
		
		
	
	} catch(Exception e) 
	{
		alert_window("Aw nah! Your webcam is missing. Please attach a webcam and restart the program.");
	}
}

void captureEvent(Capture webcam) 
{
	try
	{
		if(webcam.available()) 
		{ 
			webcam.read();
                        PImage webcam_image = webcam.get(0, 0, 640, 480);
      
                        smooth();
                        image(webcam_image, spacing, 150 + spacing, 640, 480);
                        noSmooth();
          
			
			if(record_video)
			{
				int millis_elapsed = millis() - capture_millis_last;
				time_trial_current = millis() - trial_time_start;
				
				if (millis_elapsed > frame_interval)
				{	
					//all of the trial_vars are set in the DERT app in write_server()
					String frame_filename = data_path + "images/" + 
										trial_vars +
										"=ms_" + time_trial_current + 
										".bmp";
					
					webcam_image.save(frame_filename);
					
					//println("frame interval " + frame_interval + " elapsed " + millis_elapsed);
					alert_window("recording - trial: " + trial + "\tfile: " + frame_filename);
					//println(channel_trigger);
					
					capture_millis_last = millis();
				}
				
			} else { 
				alert_window("not recording \ntrial: " + trial);
			}
			
		}
	
	} catch(Exception e) 
	{
		println("capture error!!");
	}
}
