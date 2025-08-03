package com.example.p20250107_3;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.os.Bundle;
import android.view.MotionEvent;
import android.view.View;
import android.widget.FrameLayout;
import android.widget.TextView;

import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

public class MainActivity extends AppCompatActivity {

    int flag=1;
    int[][] record=new int[40][40];

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        //lines
        makelines mk1=new makelines(this);
        FrameLayout fl1 = (FrameLayout) findViewById(R.id.fl1);
        fl1.addView(mk1);
        frametouch ft =new frametouch(this);
        fl1.setOnTouchListener(ft);
    }
    class frametouch implements View.OnTouchListener{
        Context this1;
        frametouch(Context this1){
            this.this1=this1;
        }

        @Override
        public boolean onTouch(View v, MotionEvent event) {
            FrameLayout fl1 = (FrameLayout) findViewById(R.id.fl1);

            if (event.getAction() == 0) {
                if (record[(Math.round((event.getX()) * 0.01f) * 100) / 100][(Math.round((event.getY()) * 0.01f) * 100) / 100 ] == 0) {
                    TextView tv1 = (TextView) findViewById(R.id.tv1);
                    TextView tv2 = (TextView) findViewById(R.id.tv2);
                    tv1.setText(Math.round((event.getX()) * 0.01f) * 100+" , "+Math.round((event.getY()) * 0.01f) * 100);
                    tv2.setText(((Math.round((event.getX()) * 0.01f) * 100) / 100 )+" , "+((Math.round((event.getY()) * 0.01f) * 100) / 100 ));
                    stone stone1 = new stone(this1);
                    fl1.addView(stone1);

                    stone1.setX(Math.round((event.getX()) * 0.01f) * 100 - 50);
                    stone1.setY(Math.round((event.getY()) * 0.01f) * 100 - 50);
                    flag *= -1;

                    if (flag < 0) {//white
                        record[(Math.round((event.getX()) * 0.01f) * 100) / 100][(Math.round((event.getY()) * 0.01f) * 100) / 100 ] = 1;
                    } else {
                        record[(Math.round((event.getX()) * 0.01f) * 100) / 100][(Math.round((event.getY()) * 0.01f) * 100) / 100 ] = 2;
                    }
                    searchFive(record);


                }
            }
            return true;
        }
    }
    class stone extends View{

        public stone(Context context) {
            super(context);
        }
        public void onDraw(Canvas cv){
            Paint pt = new Paint();
            if (flag<0){
                pt.setColor(Color.WHITE);
            }else{
                pt.setColor(Color.BLACK);
            }
            pt.setStyle(Paint.Style.FILL);
            pt.setStrokeWidth(1f);

            cv.drawCircle(50,50,50,pt);
        }
    }

    class makelines extends View{
        public makelines(Context context) {
            super(context);
        }
        public void onDraw(Canvas cv){
            Paint pt = new Paint();
            pt.setColor(Color.BLACK);
            pt.setStrokeWidth(5f);
            for(int i=0;i<40;i++){
                cv.drawLine(0,100*i,cv.getWidth(),100*i,pt);
                cv.drawLine(100*i,0,100*i,cv.getHeight(),pt);
            }

        }
    }

    public void searchFive(int[][] a) {
        TextView tv1 = (TextView) findViewById(R.id.tv1);
        TextView tv2 = (TextView) findViewById(R.id.tv2);
        int pos = -1;
        String str = "";

        //row
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length - 4; j++) {
                if (a[i][j] != 0 &&
                        a[i][j] == a[i][j + 1] &&
                        a[i][j] == a[i][j + 2] &&
                        a[i][j] == a[i][j + 3] &&
                        a[i][j] == a[i][j + 4]) {
                    pos = a[i][j];
                    break;
                }
            }
            if (pos != -1) break;
        }

        //column
        if (pos == -1) {
            for (int j = 0; j < a[0].length; j++) {
                for (int i = 0; i < a.length - 4; i++) {
                    if (a[i][j] != 0 &&
                            a[i][j] == a[i + 1][j] &&
                            a[i][j] == a[i + 2][j] &&
                            a[i][j] == a[i + 3][j] &&
                            a[i][j] == a[i + 4][j]) {
                        pos = a[i][j];
                        break;
                    }
                }
                if (pos != -1) break;
            }
        }

        //left up to right down
        if (pos == -1) {
            for (int i = 0; i < a.length - 4; i++) {
                for (int j = 0; j < a[0].length - 4; j++) {
                    if (a[i][j] != 0 &&
                            a[i][j] == a[i + 1][j + 1] &&
                            a[i][j] == a[i + 2][j + 2] &&
                            a[i][j] == a[i + 3][j + 3] &&
                            a[i][j] == a[i + 4][j + 4]) {
                        pos = a[i][j];
                        break;
                    }
                }
                if (pos != -1) break;
            }
        }

        //right up to left down
        if (pos == -1) {
            for (int i = 4; i < a.length; i++) {
                for (int j = 0; j < a[0].length - 4; j++) {
                    if (a[i][j] != 0 &&
                            a[i][j] == a[i - 1][j + 1] &&
                            a[i][j] == a[i - 2][j + 2] &&
                            a[i][j] == a[i - 3][j + 3] &&
                            a[i][j] == a[i - 4][j + 4]) {
                        pos = a[i][j];
                        break;
                    }
                }
                if (pos != -1) break;
            }
        }

        if (pos != -1) {
            str = (pos == 1 ? "white" : "black");
            tv1.setText(str + " win");
            tv2.setText("");
        }
    }
}