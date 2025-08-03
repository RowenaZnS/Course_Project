package com.example.afinal;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.view.MotionEvent;
import android.view.View;
import android.view.animation.AlphaAnimation;
import android.widget.FrameLayout;
import android.widget.LinearLayout;
import android.widget.TextView;
import androidx.appcompat.app.AppCompatActivity;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;




public class MainActivity extends AppCompatActivity {

    Context gthis;
    FrameLayout fl1;
    boolean which = false;
    makeball ball;
    makebar bar;
    float ball_vel_x, ball_vel_y;
    float ref_x, bar_vel;
    float counter = 0f;
    float starttime,endtime;
    float startx,endx,starty,endy;

    int score = 0;

    List<makegreenblock> greenBlocks = new ArrayList<>();

    Handler handle = new Handler(Looper.myLooper());

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        LinearLayout ll = (LinearLayout) findViewById(R.id.ll1);

        gthis = this;

        fl1 = (FrameLayout) findViewById(R.id.fl1);
        frametouch ft = new frametouch();
        fl1.setOnTouchListener(ft);

        Random random = new Random();

        TextView tv = (TextView) findViewById(R.id.tv1);
        tv.setText("" + score);

        for (int i = 0; i < 5; i++) {
            makegreenblock greenBlock = new makegreenblock(this, i + 1);
            greenBlock.setId(i);
            fl1.addView(greenBlock, 100, 100);
            greenBlock.setX(random.nextInt(1000 - 100));
            greenBlock.setY(random.nextInt(1000 - 100));
            greenBlocks.add(greenBlock);
        }

        bar_vel = (float) (Math.random() * 15 - 5);

        if (Math.abs(bar_vel) < 4) {
            bar_vel = bar_vel < 0 ? -4 : 4; // Ensure not too slow
        }


        Thread thr = new Thread("test") {
            @Override
            public void run() {
                for (;;) {
                    counter += 0.1f;
                    handle.post(new Runnable() {
                        @Override
                        public void run() {  // Running on main thread
                            if (which) {
                                // Ball movement
                                ball.setX(ball.getX() + ball_vel_x);
                                ball.setY(ball.getY() + ball_vel_y);

                                // Ball collision with walls
                                if (ball.getX() + 100f > fl1.getWidth() || ball.getX() < 0) {
                                    ball_vel_x *= -1;
                                }
                                if (ball.getY() < 0) {
                                    ball_vel_y *= -1;
                                }

                                // Ball collision with bar
                                if (ball.getY() + 100f >= bar.getY() && ball.getY() <= bar.getY() + 20 &&
                                        ball.getX() + 100f >= bar.getX() && ball.getX() <= bar.getX() + 200) {
                                    ball_vel_y *= -1; // Reverse vertical direction
                                }

                                // Check if ball touches bottom
                                if (ball.getY() + 100f >= fl1.getHeight()) {

                                    //animateAndRemove(bar);
                                    fl1.removeView(bar); // Remove bar
                                    // Optionally, remove ball or reset the game state
                                }

                                // Bar movement
                                bar.setX(bar.getX() + bar_vel);

                                // Bar collision with walls
                                if (bar.getX() < 0) {
                                    bar.setX(0);
                                    bar_vel *= -1; // Reverse
                                } else if (bar.getX() + bar.getWidth() > fl1.getWidth()) {
                                    bar.setX(fl1.getWidth() - bar.getWidth());
                                    bar_vel *= -1;
                                }

                                for (int i = 0; i < greenBlocks.size(); i++) {
                                    makegreenblock greenBlock = greenBlocks.get(i);
                                    greenBlock.setX(greenBlock.getX() + greenBlock.vel_x);

                                    // Green block collision with walls
                                    if (greenBlock.getX() + 100 > fl1.getWidth() || greenBlock.getX() < 0) {
                                        greenBlock.vel_x *= -1;
                                    }

                                    // Green block collision with ball
                                    if (ball.getX() + 100 >= greenBlock.getX() && ball.getX() <= greenBlock.getX() + 100 &&
                                            ball.getY() + 100 >= greenBlock.getY() && ball.getY() <= greenBlock.getY() + 100) {
                                        ball_vel_x *= -1;
                                        ball_vel_y *= -1;
                                        score += greenBlock.number;
                                        tv.setText("" + score);
                                        //fl1.removeView(greenBlock);
                                        greenBlocks.remove(i);
                                        animateAndRemove(greenBlock);
                                        i--;
                                    }
                                }
                            }
                        }
                    });

                    try {
                        Thread.sleep(10);
                    } catch (InterruptedException ie) {
                        ie.printStackTrace();
                    }
                }
            }
        };
        thr.start();
    }

    private void animateAndRemove(View view) {
        AlphaAnimation animation = new AlphaAnimation(1.0f, 0.0f);
        animation.setDuration(500);
        animation.setAnimationListener(new android.view.animation.Animation.AnimationListener() {
            @Override
            public void onAnimationStart(android.view.animation.Animation animation) {}
            @Override
            public void onAnimationEnd(android.view.animation.Animation animation) {
                fl1.removeView(view);
            }
            @Override
            public void onAnimationRepeat(android.view.animation.Animation animation) {}
        });
        view.startAnimation(animation);
    }

    class frametouch implements View.OnTouchListener {
        @Override
        public boolean onTouch(View v, MotionEvent ev) {
            if (ev.getAction() == 0) {
                startx = ev.getX();
                starty = ev.getY();
                starttime = counter;



                if (which == false) {


                    bar = new makebar();
                    fl1.addView(bar, 200, 20);
                    bar.setX(fl1.getWidth() / 2 - 100);
                    bar.setY(fl1.getHeight() - 30);

                    ball = new makeball();
                    fl1.addView(ball, 100, 100);
                    ball.setX(ev.getX() - 50f);
                    ball.setY(ev.getY() - 50f);
                    which = true;
                }
                ref_x = ev.getX();
            }
            if (ev.getAction() ==1) {
                endx = ev.getX()-startx;
                endy = ev.getY()-starty;
                endtime = counter-starttime;

                ball_vel_x = endx/endtime/10;
                ball_vel_y = endy/endtime/10;

            }

//            if (ev.getAction() == 2) { // Move blue bar
//                bar_vel = (ev.getX() - ref_x) * 0.1f;
//            }
//            bar_vel = (float) (Math.random() * 15 - 5);
//
//            if (Math.abs(bar_vel) < 4) {
//                bar_vel = bar_vel < 0 ? -4 : 4; // Ensure not too slow
//            }

            return true;
        }
    }

    class makeball extends View {
        makeball() {
            super(gthis);
        }

        public void onDraw(Canvas cv) {
            Paint pt = new Paint();
            pt.setColor(Color.YELLOW);
            pt.setStyle(Paint.Style.FILL);
            cv.drawCircle(50, 50, 50, pt);
        }
    }

    class makebar extends View {
        makebar() {
            super(gthis);
        }

        public void onDraw(Canvas cv) {
            Paint pt = new Paint();
            pt.setColor(Color.BLUE);
            pt.setStyle(Paint.Style.FILL);
            cv.drawRect(0, 0, 300, 40, pt);
        }
    }

    class makegreenblock extends View {
        int number;
        float vel_x;

        makegreenblock(Context context, int number) {
            super(context);
            this.number = number;
            this.vel_x = (float) (Math.random() * 10 - 5);

            if (Math.abs(this.vel_x) < 2) {
                this.vel_x = this.vel_x < 0 ? -2 : 2; // Ensure not too slow
            }
        }

        @Override
        protected void onDraw(Canvas canvas) {
            Paint paint = new Paint();
            paint.setColor(Color.GREEN);
            paint.setStyle(Paint.Style.FILL);
            canvas.drawRect(0, 0, 300, 300, paint);


            paint.setColor(Color.WHITE);
            paint.setTextSize(40);
            paint.setTextAlign(Paint.Align.CENTER);
            canvas.drawText(String.valueOf(number), 50, 60, paint);
        }
    }
}



