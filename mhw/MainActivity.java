package com.example.mobilehw1;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.os.Bundle;
import android.view.MotionEvent;
import android.view.View;
import android.widget.Button;
import android.widget.FrameLayout;

import androidx.appcompat.app.AppCompatActivity;

//makeview mv = new makeview(this) should contain the this of public class
public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        FrameLayout frame = (FrameLayout) findViewById(R.id.fl1);
        touchListener tl = new touchListener(this);
        frame.setOnTouchListener(tl);

        Button redButton = findViewById(R.id.red);
        Button greenButton = findViewById(R.id.green);
        Button blueButton = findViewById(R.id.blue);
        Button clearButton = findViewById(R.id.clear);
        redButton.setOnClickListener(new clickListener());
        greenButton.setOnClickListener(new clickListener());
        blueButton.setOnClickListener(new clickListener());
        clearButton.setOnClickListener(new clickListener());
    }

    float x=0f;
    float y=0f;
    makeview mv;
    String color="white";

    class clickListener implements View.OnClickListener{
        @Override
        public void onClick(View v) {
            if (R.id.red == v.getId()){
                color = "red";
            }else if (R.id.green == v.getId()) {
                color = "green";
            }else if (R.id.blue == v.getId()) {
                color = "blue";
            }else if (R.id.clear == v.getId()) {
                color = "white";
                clear();
            }

        }

        private void clear(){
            FrameLayout f1 = (FrameLayout) findViewById(R.id.fl1);
            f1.removeAllViews();
        }


    }

    class touchListener implements View.OnTouchListener{
        Context this1;
        touchListener(Context this1){//itself do not have constructor
            this.this1 = this1;
        }
        @Override
        public boolean onTouch(View v, MotionEvent event) {
            if (event.getAction() == 0 || event.getAction()== 1 ||event.getAction() == 2) {
                FrameLayout frame = (FrameLayout) findViewById(R.id.fl1);
                x=event.getX();
                y=event.getY();
                makeview mv = new makeview(this1);
                frame.addView(mv);
            }return true;
        }
    }
    class makeview extends View{
        makeview(Context this1){
            super(this1);
        }

        public void onDraw(Canvas cv){
            Paint pt = new Paint();
            if (color.equals("red")){
                pt.setColor(Color.RED);
            }else if (color.equals("green")){
                pt.setColor(Color.GREEN);
            }else if (color.equals("blue")){
                pt.setColor(Color.BLUE);
            } else if (color.equals("white")){
                pt.setColor(Color.WHITE);
            }
            pt.setStrokeWidth(10f);
            pt.setStrokeCap(Paint.Cap.ROUND);
            pt.setStyle(Paint.Style.STROKE);
            cv.drawPoint(x,y,pt);
        }
    }
}