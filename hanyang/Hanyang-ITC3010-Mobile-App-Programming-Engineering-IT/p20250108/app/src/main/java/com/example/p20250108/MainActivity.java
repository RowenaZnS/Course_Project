
package com.example.p20250108;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.view.MotionEvent;
import android.view.View;
import android.widget.FrameLayout;
import android.widget.TextView;



import androidx.appcompat.app.AppCompatActivity;

import java.util.Random;

public class MainActivity extends AppCompatActivity {
    float counter = 0f;
    float velx, vely,velrec;
    Handler handler = new Handler(Looper.getMainLooper());
    Random random = new Random(); // Create a Random instance
    float starttime,endtime;
    float startx,endx,starty,endy;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        makeball redball = new makeball(this);
        makeRe re = new makeRe(this);
        FrameLayout fl1 = findViewById(R.id.fl1);
        fl1.addView(redball);
        fl1.addView(re);
        TextView tv1 = findViewById(R.id.tv1);

        // Set random initial position
        redball.setX(0); // Width - ball diameter to stay in bounds
        redball.setY(0); // Height - ball diameter to stay in bounds

        re.setX(500);
        re.setY(2200);



        velrec = (random.nextFloat() * 50);

        // Activate job thread for ball movement
        Thread thr = new Thread("temp") {
            @Override
            public void run() {
                for (;;) { // Infinite loop
                    counter++;
                    handler.post(new Runnable() {
                        @Override
                        public void run() { // Run on main thread

                            redball.setX(redball.getX() + velx);
                            redball.setY(redball.getY() + vely);

                            //tv1.setText("Position: "+redball.getX()+","+redball.getY());

                            // Bounce off walls
                            if (redball.getX() >= fl1.getWidth() - 100 || redball.getX() <= 0) {
                                velx = -velx; // Reverse direction
                            }
                            if (redball.getY() >= fl1.getHeight() - 100 || redball.getY() <= 0) {
                                vely = -vely; // Reverse direction
                            }
                            redball.setX(redball.getX() + velx);
                            redball.setY(redball.getY() + vely);
                            //tv1.setText("Position: "+redball.getX()+","+redball.getY());

                            // Bounce off walls
                            if (re.getX() >= fl1.getWidth() - 200 || re.getX() <= 0) {
                                velrec = -velrec; // Reverse direction
                            }
                            re.setX(re.getX() + velrec);
//                            touchListener touchListener = new touchListener();
//                            re.setOnTouchListener(touchListener);

                            boolean collde = collde(redball,re);

                            if (collde) {
                                velx = -velx;
                                vely = -vely;
                                tv1.setText("Ball Received");
                                new Handler().postDelayed(new Runnable() {
                                    @Override
                                    public void run() {
                                        tv1.setText("");
                                    }
                                }, 1000);


                            } else if (!collde && redball.getY() >= fl1.getHeight() - 100) {
                                tv1.setText("Ball Lost");
                                new Handler().postDelayed(new Runnable() {
                                    @Override
                                    public void run() {
                                        tv1.setText("");
                                    }
                                }, 1000);
                            }

                        }
                    });
                    try {
                        Thread.sleep(10);
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                }
            }
        };
        thr.start();
        frametouch ft = new frametouch();
        fl1.setOnTouchListener(ft);
    }

    class frametouch implements View.OnTouchListener{
        public boolean onTouch(View v, MotionEvent event){
            TextView tv = (TextView) findViewById(R.id.tv1);
            if (event.getAction() ==0) {
                startx = event.getX();
                starty = event.getY();
                starttime = counter;
            }
            if (event.getAction() ==1) {
                endx = event.getX()-startx;
                endy = event.getY()-starty;
                endtime = counter-starttime;

                velx = endx/endtime;
                vely = endy/endtime;
                //tv.setText(velx+"    "+vely);
            }
            return true;
        }
    }

    public boolean run(View v, MotionEvent event) {
        // Removed touch handling since it's no longer necessary
        return true;
    }

    class makeball extends View {
        makeball(Context context) {
            super(context);
        }

        @Override
        public void onDraw(Canvas cv) {
            Paint pt = new Paint();
            pt.setColor(Color.parseColor("#00ff00"));
            pt.setStrokeWidth(30f);
            pt.setStyle(Paint.Style.FILL);
            // Draw the ball at its current position
            cv.drawCircle(getX() + 50, getY() + 50, 50, pt); // Adjust for the ball's radius
        }
    }

    class makeRe extends View {
        makeRe(Context context) {
            super(context);
        }

        @Override
        public void onDraw(Canvas cv) {
            Paint pt = new Paint();
            pt.setColor(Color.parseColor("#0000ff"));
            pt.setStrokeWidth(30f);
            pt.setStyle(Paint.Style.FILL);
            // Draw the ball at its current position
            cv.drawRect(0,0,200,100, pt);
        }
    }


//    class touchListener implements View.OnTouchListener {
//        @Override
//        public boolean onTouch(View v, MotionEvent event) {
//            if (event.getAction() == MotionEvent.ACTION_MOVE) {
//                float newX = event.getRawX();
//                v.setX(newX);
//            }
//            return true;
//        }
//    }
    public boolean collde(makeball ball, makeRe re){
        float ballX = ball.getX() + 50; //center
        float ballY = ball.getY() + 50;
        float ballRadius = 50;

        float rectX = re.getX();
        float rectY = re.getY();
        float rectWidth = 200;
        float rectHeight = 100;
        return (ballX+ballRadius >= rectX && ballX-ballRadius <= rectX + rectWidth && ballY+ballRadius >= rectY && ballY-ballRadius <= rectY + rectHeight);

    }


}
