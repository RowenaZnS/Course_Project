package com.example.midtestq2;

import android.graphics.Color;
import android.os.Bundle;
import android.view.Gravity;
import android.view.MotionEvent;
import android.view.View;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.LinearLayout.LayoutParams;
import androidx.appcompat.app.AppCompatActivity;

public class MainActivity extends AppCompatActivity {
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        LinearLayout rootLayout = new LinearLayout(this);
        rootLayout.setGravity(Gravity.CENTER);
        rootLayout.setOrientation(LinearLayout.VERTICAL);
        rootLayout.setBackgroundColor(Color.parseColor("#FFFFFF"));
        rootLayout.setLayoutParams(new LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.MATCH_PARENT));


        int[][] textSizes = {
                {50, 45, 40, 35},
                {45, 40, 35, 30},
                {40, 35, 30, 25},
                {35, 30, 25, 20}
        };

        LinearLayout[] lilist = new LinearLayout[4];
        TextView[] tvlist = new TextView[16];
        LinearLayout.LayoutParams size1 = new LayoutParams(200,200);
        size1.setMargins(5, 5, 5, 5);
        MyTouchListener tc = new MyTouchListener();

        for(int i=0; i<4; i++) {
            lilist[i] = new LinearLayout(this);
            lilist[i].setOrientation(LinearLayout.HORIZONTAL);
            lilist[i].setBackgroundColor(Color.parseColor("#FFFFFF"));
            lilist[i].setLayoutParams(new LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.WRAP_CONTENT));
            lilist[i].setGravity(Gravity.CENTER);
            rootLayout.addView(lilist[i]);

            for (int j = 0; j < 4; j++) {
                tvlist[i * 4 + j] = new TextView(this);
                tvlist[i * 4 + j].setTextSize(textSizes[i][j]);
                tvlist[i * 4 + j].setText("" + textSizes[i][j]);
                tvlist[i * 4 + j].setGravity(Gravity.CENTER);
                tvlist[i * 4 + j].setPadding(10, 10, 10, 10);
                tvlist[i * 4 + j].setBackgroundColor(Color.parseColor("#00BCD4"));
                tvlist[i * 4 + j].setTextColor(Color.parseColor("#000000"));
                tvlist[i * 4 + j].setLayoutParams(size1);
                tvlist[i * 4 + j].setGravity(Gravity.CENTER);
                tvlist[i * 4 + j].setId((int) (i * 4 + j));
                tvlist[i * 4 + j].setOnTouchListener(tc);
                lilist[i].addView(tvlist[i * 4 + j]);
            }
        }

        setContentView(rootLayout);
    }

    class MyTouchListener implements View.OnTouchListener {
        @Override
        public boolean onTouch(View v, MotionEvent event) {
            if (v.getId() == (int) 0 ) {
                if (event.getAction() == 0) {
                    v.setBackgroundColor(Color.parseColor("#E91E63"));
                    ((TextView) v).setTextColor(Color.parseColor("#FFFFFF"));
                } else if (event.getAction() == 1) {
                    v.setBackgroundColor(Color.parseColor("#00BCD4"));
                    ((TextView) v).setTextColor(Color.parseColor("#000000"));
                }
            }else if (v.getId() == (int) 1 ) {
                if (event.getAction() == 0){
                    v.setBackgroundColor(Color.parseColor("#E91E63"));
                    ((TextView) v).setTextColor(Color.parseColor("#FFFFFF"));
                }else if (event.getAction() == 1){
                    v.setBackgroundColor(Color.parseColor("#00BCD4"));
                    ((TextView) v).setTextColor(Color.parseColor("#000000"));
                }
            }else if (v.getId() == (int) 2 ) {
                if (event.getAction() == 0){
                    v.setBackgroundColor(Color.parseColor("#E91E63"));
                    ((TextView) v).setTextColor(Color.parseColor("#FFFFFF"));
                }else if (event.getAction() == 1){
                    v.setBackgroundColor(Color.parseColor("#00BCD4"));
                    ((TextView) v).setTextColor(Color.parseColor("#000000"));
                }
            }
            else if (v.getId() == (int) 3 ) {
                if (event.getAction() == 0){
                    v.setBackgroundColor(Color.parseColor("#E91E63"));
                    ((TextView) v).setTextColor(Color.parseColor("#FFFFFF"));
                }else if (event.getAction() == 1){
                    v.setBackgroundColor(Color.parseColor("#00BCD4"));
                    ((TextView) v).setTextColor(Color.parseColor("#000000"));
                }
            }
            else if (v.getId() == (int) 4 ) {
                if (event.getAction() == 0){
                    v.setBackgroundColor(Color.parseColor("#E91E63"));
                    ((TextView) v).setTextColor(Color.parseColor("#FFFFFF"));
                }else if (event.getAction() == 1){
                    v.setBackgroundColor(Color.parseColor("#00BCD4"));
                    ((TextView) v).setTextColor(Color.parseColor("#000000"));
                }
            }
            else if (v.getId() == (int) 5 ) {
                if (event.getAction() == 0){
                    v.setBackgroundColor(Color.parseColor("#E91E63"));
                    ((TextView) v).setTextColor(Color.parseColor("#FFFFFF"));
                }else if (event.getAction() == 1){
                    v.setBackgroundColor(Color.parseColor("#00BCD4"));
                    ((TextView) v).setTextColor(Color.parseColor("#000000"));
                }
            }
            else if (v.getId() == (int) 6 ) {
                if (event.getAction() == 0){
                    v.setBackgroundColor(Color.parseColor("#E91E63"));
                    ((TextView) v).setTextColor(Color.parseColor("#FFFFFF"));
                }else if (event.getAction() == 1){
                    v.setBackgroundColor(Color.parseColor("#00BCD4"));
                    ((TextView) v).setTextColor(Color.parseColor("#000000"));
                }
            }
            else if (v.getId() == (int) 7 ) {
                if (event.getAction() == 0){
                    v.setBackgroundColor(Color.parseColor("#E91E63"));
                    ((TextView) v).setTextColor(Color.parseColor("#FFFFFF"));
                }else if (event.getAction() == 1){
                    v.setBackgroundColor(Color.parseColor("#00BCD4"));
                    ((TextView) v).setTextColor(Color.parseColor("#000000"));
                }
            }
            else if (v.getId() == (int) 8 ) {
                if (event.getAction() == 0){
                    v.setBackgroundColor(Color.parseColor("#E91E63"));
                    ((TextView) v).setTextColor(Color.parseColor("#FFFFFF"));
                }else if (event.getAction() == 1){
                    v.setBackgroundColor(Color.parseColor("#00BCD4"));
                    ((TextView) v).setTextColor(Color.parseColor("#000000"));
                }
            }
            else if (v.getId() == (int) 9 ) {
                if (event.getAction() == 0){
                    v.setBackgroundColor(Color.parseColor("#E91E63"));
                    ((TextView) v).setTextColor(Color.parseColor("#FFFFFF"));
                }else if (event.getAction() == 1){
                    v.setBackgroundColor(Color.parseColor("#00BCD4"));
                    ((TextView) v).setTextColor(Color.parseColor("#000000"));
                }
            }
            else if (v.getId() == (int) 10 ) {
                if (event.getAction() == 0){
                    v.setBackgroundColor(Color.parseColor("#E91E63"));
                    ((TextView) v).setTextColor(Color.parseColor("#FFFFFF"));
                }else if (event.getAction() == 1){
                    v.setBackgroundColor(Color.parseColor("#00BCD4"));
                    ((TextView) v).setTextColor(Color.parseColor("#000000"));
                }
            }
            else if (v.getId() == (int) 11 ) {
                if (event.getAction() == 0){
                    v.setBackgroundColor(Color.parseColor("#E91E63"));
                    ((TextView) v).setTextColor(Color.parseColor("#FFFFFF"));
                }else if (event.getAction() == 1){
                    v.setBackgroundColor(Color.parseColor("#00BCD4"));
                    ((TextView) v).setTextColor(Color.parseColor("#000000"));
                }
            }
            else if (v.getId() == (int) 12 ) {
                if (event.getAction() == 0){
                    v.setBackgroundColor(Color.parseColor("#E91E63"));
                    ((TextView) v).setTextColor(Color.parseColor("#FFFFFF"));
                }else if (event.getAction() == 1){
                    v.setBackgroundColor(Color.parseColor("#00BCD4"));
                    ((TextView) v).setTextColor(Color.parseColor("#000000"));
                }
            }
            else if (v.getId() == (int) 13 ) {
                if (event.getAction() == 0){
                    v.setBackgroundColor(Color.parseColor("#E91E63"));
                    ((TextView) v).setTextColor(Color.parseColor("#FFFFFF"));
                }else if (event.getAction() == 1){
                    v.setBackgroundColor(Color.parseColor("#00BCD4"));
                    ((TextView) v).setTextColor(Color.parseColor("#000000"));
                }
            }
            else if (v.getId() == (int) 14 ) {
                if (event.getAction() == 0){
                    v.setBackgroundColor(Color.parseColor("#E91E63"));
                    ((TextView) v).setTextColor(Color.parseColor("#FFFFFF"));
                }else if (event.getAction() == 1){
                    v.setBackgroundColor(Color.parseColor("#00BCD4"));
                    ((TextView) v).setTextColor(Color.parseColor("#000000"));
                }
            }
            else if (v.getId() == (int) 15 ) {
                if (event.getAction() == 0){
                    v.setBackgroundColor(Color.parseColor("#E91E63"));
                    ((TextView) v).setTextColor(Color.parseColor("#FFFFFF"));
                }else if (event.getAction() == 1){
                    v.setBackgroundColor(Color.parseColor("#00BCD4"));
                    ((TextView) v).setTextColor(Color.parseColor("#000000"));
                }
            }
            return true;
        }
    }
}