/* 
 * Commands: 
 * O = opto on; command = O, frequency, width, duration
 * X = opto off
 * R = reboot
 */


const int opto_pin = 0;
unsigned int opto_start = 0,
  opto_duty_cycle = 0,
  opto_freq = 0,
  opto_width = 0,
  opto_dur = 0;

unsigned int read_int16() {
  union u_tag { 
    byte b[2];
    unsigned int val;
  } par;
  for (int i=0; i<2; i++){
    if ((Serial.available() > 0))
      par.b[i] = Serial.read();
    else
      par.b[i] = 0;
  }
  return par.val;
}

void setup() {
  Serial.begin(115200);
  pinMode(opto_pin, OUTPUT);
}

void loop() {

  unsigned int curr_time = millis();
  
  while (Serial.available() > 0) {

    unsigned int cmd = Serial.read();
    
    if(cmd == 'O') {
      
      opto_start = curr_time;
      opto_freq = read_int16();
      opto_width = read_int16();
      opto_dur = read_int16();
      if (opto_dur == 0)
        opto_dur = 65355;
      opto_duty_cycle = opto_width * opto_freq * 4096 / 1000;
      analogWriteFrequency(opto_pin, opto_freq);
      analogWrite(opto_pin, opto_duty_cycle);

      Serial.print(opto_freq);
      Serial.print(',');
      Serial.print(opto_width);
      Serial.print(',');
      Serial.print(opto_dur);
      Serial.print('\n');
      Serial.flush();
      
    } else if(cmd == 'X') {

      analogWrite(opto_pin, 0);
      
    } else if(cmd == 'R') {

      _reboot_Teensyduino_();
      
    }
  }

  if (curr_time > opto_start + opto_dur)
    analogWrite(opto_pin, 0);

}
