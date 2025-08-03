package com.example.midtest1;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.os.Bundle;
import android.util.AttributeSet;
import android.view.Gravity;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup;
import android.widget.FrameLayout;
import android.widget.LinearLayout;
import android.widget.TextView;

import androidx.activity.EdgeToEdge;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

//makeview mv = new makeview(this) should contain the this of public class
public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        FrameLayout frame = (FrameLayout) findViewById(R.id.fl1);
        touchListener tl = new touchListener(this);
        frame.setOnTouchListener(tl);
    }

    float x=0f;
    float y=0f;
    makeview mv;

    class touchListener implements View.OnTouchListener{
        Context this1;
        touchListener(Context this1){//itself do not have constructor
            this.this1 = this1;
        }
        @Override
        public boolean onTouch(View v, MotionEvent event) {
            if (event.getAction() == 0){
                FrameLayout frame = (FrameLayout) findViewById(R.id.fl1);
                x=event.getX();
                y=event.getY();
                makeview mv = new makeview(this1);//MainActivity.this also represent
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
            pt.setColor(Color.RED);
            pt.setStrokeWidth(10f);
            pt.setStrokeCap(Paint.Cap.ROUND);
            pt.setStyle(Paint.Style.STROKE);
            cv.drawPoint(x,y,pt);
            //cv.drawCircle(200,200,100,pt);
            //pt.setColor(Color.GREEN);
            //cv.drawLine(0,0,300,300,pt);
        }
    }
}