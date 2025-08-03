package com.example.day4;

import android.graphics.Color;
import android.os.Bundle;
import android.view.Gravity;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.TextView;

import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

import org.w3c.dom.Text;

public class MainActivity extends AppCompatActivity {
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        LinearLayout.LayoutParams size = new LinearLayout.LayoutParams(200,200);
        size.setMargins(10,10,10,10);
        LinearLayout linear = (LinearLayout) findViewById(R.id.ll1) ;

        LinearLayout l1 = new LinearLayout(this);
        l1.setOrientation(LinearLayout.HORIZONTAL);
        l1.setGravity(Gravity.CENTER);
        LinearLayout l2 = new LinearLayout(this);
        l2.setOrientation(LinearLayout.HORIZONTAL);
        l2.setGravity(Gravity.CENTER);
        LinearLayout l3 = new LinearLayout(this);
        l3.setOrientation(LinearLayout.HORIZONTAL);
        l3.setGravity(Gravity.CENTER);
        LinearLayout l4 = new LinearLayout(this);
        l4.setOrientation(LinearLayout.HORIZONTAL);
        l4.setGravity(Gravity.CENTER);
        TextView[] tvlist = new TextView[12];
        for (int i = 1; i <=3; i++) {
            tvlist[i] = new TextView(this);
            tvlist[i].setTextSize(50f);
            tvlist[i].setTextColor(Color.BLACK);
            tvlist[i].setBackgroundColor(Color.parseColor("#ff9800"));
            tvlist[i].setGravity(Gravity.CENTER);
            tvlist[i].setLayoutParams(size);
            tvlist[i].setText(""+i);
            tvlist[i].setId((int)i);
            l1.addView(tvlist[i]);
        }
        linear.addView(l1);

        for (int i = 4; i <=6; i++) {
            tvlist[i] = new TextView(this);
            tvlist[i].setTextSize(50f);
            tvlist[i].setTextColor(Color.BLACK);
            tvlist[i].setBackgroundColor(Color.parseColor("#ff9800"));
            tvlist[i].setGravity(Gravity.CENTER);
            tvlist[i].setLayoutParams(size);
            tvlist[i].setText(""+i);
            tvlist[i].setId((int)i);
            l2.addView(tvlist[i]);
        }
        linear.addView(l2);

        for (int i = 7; i <=9; i++) {
            tvlist[i] = new TextView(this);
            tvlist[i].setTextSize(50f);
            tvlist[i].setTextColor(Color.BLACK);
            tvlist[i].setBackgroundColor(Color.parseColor("#ff9800"));
            tvlist[i].setGravity(Gravity.CENTER);
            tvlist[i].setLayoutParams(size);
            tvlist[i].setText(""+i);
            tvlist[i].setId((int)i);
            l3.addView(tvlist[i]);
        }
        linear.addView(l3);
        tvlist[10] = new TextView(this);
        tvlist[10].setTextSize(50f);
        tvlist[10].setTextColor(Color.BLACK);
        tvlist[10].setBackgroundColor(Color.parseColor("#ff9800"));
        tvlist[10].setGravity(Gravity.CENTER);
        tvlist[10].setLayoutParams(size);
        tvlist[10].setText("+");
        tvlist[10].setId((int)10);
        l4.addView(tvlist[10]);

        tvlist[0] = new TextView(this);
        tvlist[0].setTextSize(50f);
        tvlist[0].setTextColor(Color.BLACK);
        tvlist[0].setBackgroundColor(Color.parseColor("#ff9800"));
        tvlist[0].setGravity(Gravity.CENTER);
        tvlist[0].setLayoutParams(size);
        tvlist[0].setText("0");
        tvlist[0].setId((int)0);
        l4.addView(tvlist[0]);

        tvlist[11] = new TextView(this);
        tvlist[11].setTextSize(50f);
        tvlist[11].setTextColor(Color.BLACK);
        tvlist[11].setBackgroundColor(Color.parseColor("#ff9800"));
        tvlist[11].setGravity(Gravity.CENTER);
        tvlist[11].setLayoutParams(size);
        tvlist[11].setText("=");
        tvlist[11].setId((int)11);
        l4.addView(tvlist[11]);
        linear.addView(l4);

        buttonclick btc = new buttonclick();
        for (int i = 0; i < tvlist.length; i++) {
            tvlist[i].setOnClickListener(btc);
        }

    }
    String num = "";
    class buttonclick implements View.OnClickListener{
        @Override
        public void onClick(View v){
            TextView screen = (TextView) findViewById(R.id.screen);
            if (v.getId()==(int) 0){
                num=num+0;
                screen.setText(num);
            }
            if (v.getId()==(int) 1){
                num=num+1;
                screen.setText(num);
            }
            if (v.getId()==(int) 2){
                num=num+2;
                screen.setText(num);
            }
            if (v.getId()==(int) 3){
                num=num+3;
                screen.setText(num);
            }
            if (v.getId()==(int) 4){
                num=num+4;
                screen.setText(num);
            }
            if (v.getId()==(int) 5){
                num=num+5;
                screen.setText(num);
            }

            if (v.getId()==(int) 6){
                num=num+6;
                screen.setText(num);
            }
            if (v.getId()==(int) 7){
                num=num+7;
                screen.setText(num);
            }
            if (v.getId()==(int) 8){
                num=num+8;
                screen.setText(num);
            }
            if (v.getId()==(int) 9){
                num=num+90;
                screen.setText(num);
            }
            if (v.getId()==(int) 10){
                num=num+"+";
                screen.setText(num);
            }

            if (v.getId()==(int) 11){
                num=num+"=";
                screen.setText(num);
            }

        }

    }

//    class buttonclick implements View.OnClickListener{
//        @Override
//        public void onClick(View v){
//            TextView screen = (TextView) findViewById(R.id.screen);
//            if (v.getId()==(int)1){
//
//            }
//
//        }
//    }
//    @Override
//    protected void onCreate(Bundle savedInstanceState) {
//        super.onCreate(savedInstanceState);
//        setContentView(R.layout.activity_main);
//        Button bt = (Button) findViewById(R.id.bt1);
//        btclick btc = new btclick();
//        bt.setOnClickListener(btc);
//    }
//    class btclick implements View.OnClickListener{
//        @Override
//        public void onClick(View v){
//            LinearLayout linear = (LinearLayout) findViewById(R.id.ll1);
//            linear.setBackgroundColor(Color.BLUE);
//        }
//
//    }

//    @Override
//    protected void onCreate(Bundle savedInstanceState) {
//        super.onCreate(savedInstanceState);
//        setContentView(R.layout.activity_main);
//
//        TextView ds = (TextView) findViewById(R.id.display);
//        TextView t1 = (TextView) findViewById(R.id.s1);
//        TextView t2 = (TextView) findViewById(R.id.s2);
//        TextView t3 = (TextView) findViewById(R.id.s3);
//        TextView t4 = (TextView) findViewById(R.id.s4);
//        TextView t5 = (TextView) findViewById(R.id.s5);
//        TextView t6 = (TextView) findViewById(R.id.s6);
//        TextView t7 = (TextView) findViewById(R.id.s7);
//        TextView t8 = (TextView) findViewById(R.id.s8);
//        TextView t9 = (TextView) findViewById(R.id.s9);
//        TextView tp = (TextView) findViewById(R.id.splus);
//        TextView t0 = (TextView) findViewById(R.id.s0);
//        TextView te = (TextView) findViewById(R.id.seq);
//        touchbox tb = new touchbox();
//        ds.setOnTouchListener(tb);
//
//        t1.setOnTouchListener(tb);
//        t2.setOnTouchListener(tb);
//        t3.setOnTouchListener(tb);
//        t4.setOnTouchListener(tb);
//        t5.setOnTouchListener(tb);
//        t6.setOnTouchListener(tb);
//        t7.setOnTouchListener(tb);
//        t8.setOnTouchListener(tb);
//        t9.setOnTouchListener(tb);
//        tp.setOnTouchListener(tb);
//        t0.setOnTouchListener(tb);
//        te.setOnTouchListener(tb);
//
//
//    }
//
//    class touchbox implements View.OnTouchListener{
//        @Override
//        public boolean onTouch(View v, MotionEvent ev){
//            TextView text = (TextView) findViewById(R.id.display);
//            if (v.getId()==R.id.s1 && ev.getAction()==0){
//                text.setText(text.getText()+"1");
//            }
//            if (v.getId()==R.id.s2 && ev.getAction()==0){
//                text.setText(text.getText()+"2");
//            }
//            if (v.getId()==R.id.s3 && ev.getAction()==0){
//                text.setText(text.getText()+"3");
//            }
//            if (v.getId()==R.id.s4 && ev.getAction()==0){
//                text.setText(text.getText()+"4");
//            }
//            if (v.getId()==R.id.s5 && ev.getAction()==0){
//                text.setText(text.getText()+"5");
//            }
//            if (v.getId()==R.id.s6 && ev.getAction()==0){
//                text.setText(text.getText()+"6");
//            }
//
//            if (v.getId()==R.id.s7 && ev.getAction()==0){
//                text.setText(text.getText()+"7");
//            }
//            if (v.getId()==R.id.s8 && ev.getAction()==0){
//                text.setText(text.getText()+"8");
//            }
//            if (v.getId()==R.id.s9 && ev.getAction()==0){
//                text.setText(text.getText()+"9");
//            }
//            if (v.getId()==R.id.splus && ev.getAction()==0){
//                text.setText(text.getText()+"+");
//            }
//            if (v.getId()==R.id.s0 && ev.getAction()==0){
//                text.setText(text.getText()+"0");
//            }
//            if (v.getId()==R.id.seq && ev.getAction()==0){
//                text.setText(text.getText()+"=");
//            }
//            return true;
//        }
//
//    }
}