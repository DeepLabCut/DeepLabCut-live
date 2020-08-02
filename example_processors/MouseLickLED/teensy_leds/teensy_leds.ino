const int LED = 0;
const int IR = 1;
const int REC = 2;

void blink() {

  Serial.write(!digitalRead(REC));
  Serial.flush();
  noTone(IR);
  while (digitalRead(REC) == 0) {}
  
}

void setup() {

  pinMode(LED, OUTPUT);
  pinMode(IR, OUTPUT);
  pinMode(REC, INPUT);
  attachInterrupt(digitalPinToInterrupt(REC), blink, FALLING);

  Serial.begin(9600);
}

void loop() {

  unsigned int ser_avail = Serial.available();
  
  while (ser_avail > 0) {
    
    unsigned int cmd = Serial.read();

    if (cmd == 'L') {
      
      digitalWrite(LED, !digitalRead(LED));
    
    } else if (cmd == 'R') {

      Serial.write(digitalRead(LED));
      Serial.flush();
      
    } else if (cmd == 'I') {

      tone(IR, 38000);
      
    }
    
  }
  
}
