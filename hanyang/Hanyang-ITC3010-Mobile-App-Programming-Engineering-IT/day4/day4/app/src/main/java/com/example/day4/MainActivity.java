package com.example.day4;

import android.annotation.SuppressLint;
import android.graphics.Color;
import android.os.Bundle;
import android.view.Gravity;
import android.view.MotionEvent;
import android.view.View;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.TextView;

import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

public class MainActivity extends AppCompatActivity {

    @SuppressLint("ClickableViewAccessibility")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
//        setContentView(R.layout.activity_main);
//        LinearLayout linear = (LinearLayout) findViewById(R.id.ll1);
//        touchlinear t1 =new touchlinear();
//        linear.setOnTouchListener(t1);
//        calculator();
        button_calculator();
    }
    @SuppressLint("ResourceType")
    void button_calculator(){
        setContentView(R.layout.java_calculator);
        LinearLayout ll1=(LinearLayout) findViewById(R.id.ll1);

        LinearLayout.LayoutParams textview_size = new
                LinearLayout.LayoutParams(200,200);
        textview_size.setMargins(10,10,10,10);
        LinearLayout.LayoutParams horzon = new
                LinearLayout.LayoutParams(LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT);
        btclick btc = new btclick();
        for(int i=1;i<=3;i++) {
            LinearLayout linear_line = new LinearLayout(this);
            linear_line.setLayoutParams(horzon);
            linear_line.setOrientation(LinearLayout.HORIZONTAL);
            linear_line.setGravity(Gravity.CENTER);
            linear_line.setBackgroundColor(Color.parseColor("#FFFFFF"));
            for (int j = 1; j <= 3; j++) {
                TextView tv = new TextView(this);
                tv.setText(((j+3*(i-1)+"")));
                tv.setTextSize(50f);
                tv.setId((int) j+3*(i-1));
                tv.setLayoutParams(textview_size);
                tv.setBackgroundColor(Color.parseColor("#FF9800"));
                tv.setGravity(Gravity.CENTER);
                tv.setOnClickListener(btc);
                linear_line.addView(tv);
            }
            ll1.addView(linear_line);
        }
        // last line
        LinearLayout linear_line = new LinearLayout(this);
        linear_line.setLayoutParams(horzon);
        linear_line.setOrientation(LinearLayout.HORIZONTAL);
        linear_line.setGravity(Gravity.CENTER);
        linear_line.setBackgroundColor(Color.parseColor("#FFFFFF"));

        TextView tv = new TextView(this);
        tv.setText("+");
        tv.setTextSize(50f);
        tv.setLayoutParams(textview_size);
        tv.setBackgroundColor(Color.parseColor("#FF9800"));
        tv.setGravity(Gravity.CENTER);
        tv.setId((int) 10);
        tv.setOnClickListener(btc);
        linear_line.addView(tv);

        tv = new TextView(this);
        tv.setText("0");
        tv.setTextSize(50f);
        tv.setLayoutParams(textview_size);
        tv.setBackgroundColor(Color.parseColor("#FF9800"));
        tv.setGravity(Gravity.CENTER);
        tv.setId((int) 11);
        tv.setOnClickListener(btc);
        linear_line.addView(tv);

        tv = new TextView(this);
        tv.setText("=");
        tv.setTextSize(50f);
        tv.setLayoutParams(textview_size);
        tv.setBackgroundColor(Color.parseColor("#FF9800"));
        tv.setGravity(Gravity.CENTER);
        tv.setId((int) 12);
        tv.setOnClickListener(btc);
        linear_line.addView(tv);

        ll1.addView(linear_line);
        // clear
        linear_line = new LinearLayout(this);
        linear_line.setLayoutParams(horzon);
        linear_line.setOrientation(LinearLayout.HORIZONTAL);
        linear_line.setGravity(Gravity.CENTER);
        linear_line.setBackgroundColor(Color.parseColor("#FFFFFF"));

        tv = new TextView(this);
        tv.setText("Clear");
        tv.setTextSize(50f);
        LinearLayout.LayoutParams clear_size = new
                LinearLayout.LayoutParams(640,200);
        textview_size.setMargins(10,10,10,10);
        tv.setLayoutParams(clear_size);
        tv.setBackgroundColor(Color.parseColor("#FF9800"));
        tv.setGravity(Gravity.CENTER);
        tv.setId((int) 13);
        tv.setOnClickListener(btc);
        linear_line.addView(tv);

        ll1.addView(linear_line);


    }
    class btclick implements View.OnClickListener{
        @Override
        public void onClick(View v){
            TextView textView= (TextView) findViewById(R.id.viewbox);
            LinearLayout linear =(LinearLayout) findViewById(R.id.ll1);
            if(v.getId()==(int) 0){
                if(!textView.getText().equals("0"))textView.setText(textView.getText()+"0");
                else textView.setText("0");
            }else if(v.getId()==(int) 1){
                if(!textView.getText().equals("0"))textView.setText(textView.getText()+"1");
                else textView.setText("1");
            }else if(v.getId()==(int) 2){
                if(!textView.getText().equals("0"))textView.setText(textView.getText()+"2");
                else textView.setText("2");
            }else if(v.getId()==(int) 3){
                if(!textView.getText().equals("0"))textView.setText(textView.getText()+"3");
                else textView.setText("3");
            }else if(v.getId()==(int) 4){
                if(!textView.getText().equals("0"))textView.setText(textView.getText()+"4");
                else textView.setText("4");
            }else if(v.getId()==(int) 5){
                if(!textView.getText().equals("0"))textView.setText(textView.getText()+"5");
                else textView.setText("5");
            }else if(v.getId()==(int) 6){
                if(!textView.getText().equals("0"))textView.setText(textView.getText()+"6");
                else textView.setText("6");
            }else if(v.getId()==(int) 7){
                if(!textView.getText().equals("0"))textView.setText(textView.getText()+"7");
                else textView.setText("7");
            }else if(v.getId()==(int) 8){
                if(!textView.getText().equals("0"))textView.setText(textView.getText()+"8");
                else textView.setText("8");
            }else if(v.getId()==(int) 9){
                if(!textView.getText().equals("0"))textView.setText(textView.getText()+"9");
                else textView.setText("9");
            }else if(v.getId()==(int) 10){
                if(!textView.getText().equals("0")&&textView.getText().charAt(textView.getText().length()-1)!='+')textView.setText(textView.getText()+"+");
//                else textView.setText("+");
            }else if(v.getId()==(int) 11){
                if(textView.getText().charAt(textView.getText().length()-1)!='+'&&!textView.getText().equals("0"))textView.setText(textView.getText()+"0");
                //else textView.setText("0");
            }
            else if(v.getId()==(int) 12){
                textView.setText(calculateSum((String) textView.getText()));
            }else if(v.getId()==(int) 13){
                textView.setText("0");
            }
        }
    }
    public static String calculateSum(String input) {
        // 检查输入是否为空或空字符串
        // 使用 "+" 分割字符串，将每个数字部分提取出来
        String[] numbers = input.split("\\+");
        int sum = 0;
        for (String number : numbers) {
            try {
                sum += Integer.parseInt(number.trim());
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException("输入字符串包含非法字符: " + number);
            }
        }
        return sum+"";
    }
    public void calculator(){
        setContentView(R.layout.linearlayout);
        TouchCalculator touchclc= new TouchCalculator();
        TextView [] container =new TextView[13];
        TouchCalculator touch= new TouchCalculator();
        container [0]= (TextView)findViewById(R.id.n0);
        container [1]= (TextView)findViewById(R.id.n1);
        container [2]= (TextView)findViewById(R.id.n2);
        container [3]= (TextView)findViewById(R.id.n3);
        container [4]= (TextView)findViewById(R.id.n4);
        container [5]= (TextView)findViewById(R.id.n5);
        container [6]= (TextView)findViewById(R.id.n6);
        container [7]= (TextView)findViewById(R.id.n7);
        container [8]= (TextView)findViewById(R.id.n8);
        container [9]= (TextView)findViewById(R.id.n9);
        container [10]= (TextView)findViewById(R.id.nplus);
        container [11]= (TextView)findViewById(R.id.n3);
        for(int i=0;i<=11;i++){
            container[i].setOnTouchListener(touchclc);
        }
    }

    public void ontouchbox(){
        LinearLayout small1 =(LinearLayout) findViewById(R.id.small1);
        LinearLayout small2 =(LinearLayout) findViewById(R.id.small2);
        touchbox tv = new touchbox();
        small1.setOnTouchListener(tv);
        small2.setOnTouchListener(tv);
    }

    @SuppressLint("ClickableViewAccessibility")
    public void ontouch(){
        LinearLayout linear = (LinearLayout) findViewById(R.id.ll1);
        linear.setOnTouchListener(new View.OnTouchListener() {
            @Override
            public boolean onTouch(View v, MotionEvent event) {
                return true;
            }
        });
    }
    class TouchCalculator implements View.OnTouchListener {

        @SuppressLint("SetTextI18n")
        @Override
        public boolean onTouch(View v, MotionEvent event) {
            TextView textbox =(TextView) findViewById(R.id.viewbox);
            if(v.getId()==R.id.n0&&event.getAction()==0){//push
                if(!textbox.getText().equals("0"))textbox.setText(textbox.getText()+"0");
                else textbox.setText("0");
            }else if(v.getId()==R.id.n1&&event.getAction()==0){//push
                if(!textbox.getText().equals("0"))textbox.setText(textbox.getText()+"1");
                else textbox.setText("1");
            }
            else if(v.getId()==R.id.n2&&event.getAction()==0){//push
                if(!textbox.getText().equals("0"))textbox.setText(textbox.getText()+"2");
                else textbox.setText("2");
            }else if(v.getId()==R.id.n3&&event.getAction()==0){//push
                if(!textbox.getText().equals("0"))textbox.setText(textbox.getText()+"3");
                else textbox.setText("3");
            }
            else if(v.getId()==R.id.n4&&event.getAction()==0){//push
                if(!textbox.getText().equals("0"))textbox.setText(textbox.getText()+"4");
                else textbox.setText("4");
            }else if(v.getId()==R.id.n5&&event.getAction()==0){//push
                if(!textbox.getText().equals("0"))textbox.setText(textbox.getText()+"5");
                else textbox.setText("5");
            }else if(v.getId()==R.id.n6&&event.getAction()==0){//push
                if(!textbox.getText().equals("0"))textbox.setText(textbox.getText()+"6");
                else textbox.setText("6");
            }
            else if(v.getId()==R.id.n7&&event.getAction()==0) {//push
                if (!textbox.getText().equals("0")) textbox.setText(textbox.getText() + "7");
                else textbox.setText("7");
            }
            return true;
        }
    }

class touchlinear implements View.OnTouchListener {

        @Override
        public boolean onTouch(View v, MotionEvent event) {
            TextView tv1= (TextView) findViewById(R.id.tv1);
            //0: push, 1: up, 2:moving
            if(event.getAction()==0) tv1.setText("push"+" X: "+event.getX()+" Y:"+ event.getY());
            if(event.getAction()==1) tv1.setText("up"+" X: "+event.getX()+" Y:"+ event.getY());
            if (event.getAction()==2) tv1.setText("moving"+" X: "+event.getX()+" Y:"+ event.getY());
            return true;
        }
    }
    class click implements View.OnClickListener{
        @SuppressLint("ResourceType")
        @Override
        public void onClick(View v){
            LinearLayout ll1 =(LinearLayout) findViewById(R.id.ll1);
            if(v.getId()==R.id.small1){
                ll1.setBackgroundColor(getColor(R.id.small1));
            }else if(v.getId()==R.id.small2){
                ll1.setBackgroundColor(getColor(R.id.small2));
            }
        }
    }
    class touchbox implements View.OnTouchListener{
        @SuppressLint("ResourceType")
        @Override
        public boolean onTouch (View v, MotionEvent event){
            LinearLayout ll1 =(LinearLayout) findViewById(R.id.ll1);
            LinearLayout small1= (LinearLayout) findViewById(R.id.small1);
            if(v.getId()==R.id.small1&&event.getAction()==0){//push
                ll1.setBackgroundColor(Color.RED);
            }else if(v.getId()==R.id.small2&&event.getAction()==0){//push
                ll1.setBackgroundColor(Color.BLUE);
            }
            else if(event.getAction()==1) ll1.setBackgroundColor(Color.WHITE);
            return true;
        }
    }
    }
