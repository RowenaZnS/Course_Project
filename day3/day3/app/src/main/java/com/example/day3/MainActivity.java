package com.example.day3;

import android.graphics.Color;
import android.os.Bundle;
import android.view.Gravity;
import android.widget.Button;
import android.widget.GridLayout;
import android.widget.LinearLayout;
import android.widget.RelativeLayout;
import android.widget.TextView;

import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AppCompatActivity;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        //color();
        //text();
        //RelativeLayout();
        xmljava();
    }
    public void xmljava(){
        setContentView(R.layout.activity_main);
        GridLayout grid= (GridLayout) findViewById(R.id.gl1);
        LinearLayout.LayoutParams btsize = new LinearLayout.LayoutParams(35, 35);
        for(int i=1;i<=900;i++){
            TextView tv = new TextView(this);
            tv.setText(i+"");
            tv.setTextSize(6f);
            tv.setLayoutParams(btsize);
            tv.setBackgroundColor(Color.parseColor("#FF9800"));
            tv.setGravity(Gravity.CENTER);
            grid.addView(tv);
        }
    }
    public void RelativeLayout(){
        RelativeLayout.LayoutParams relativepr = new
                RelativeLayout.LayoutParams(RelativeLayout.LayoutParams.WRAP_CONTENT,
                RelativeLayout.LayoutParams.WRAP_CONTENT);
        RelativeLayout relative = new RelativeLayout(this);
        relative.setLayoutParams(relativepr);
//        Button bt1 = new Button(this);
//        Button bt2 = new Button(this);
//        bt1.setText("aaa");
//        int id1 = 10;
//        bt1.setId(id1);
//        bt2.setText("bbb");
//        int id2 = 20;
//        bt2.setId(id2);
//        relativepr.addRule(RelativeLayout.RIGHT_OF, id1);
//        bt2.setLayoutParams(relativepr);
//        relative.addView(bt1);
//        relative.addView(bt2);
        RelativeLayout.LayoutParams tvsize2 = new RelativeLayout.LayoutParams(200, 200);
        RelativeLayout.LayoutParams tvsize = new RelativeLayout.LayoutParams(200, 200);
        TextView [] tv= new TextView[2];
        for(int i=0;i<2;i++){
            tv[i]=new TextView(this);
            //tv[i].setLayoutParams(tvsize);
            tv[i].setText("aa");
            tv[i].setTextSize(50f);
            tv[i].setBackgroundColor(Color.BLUE);
            tv[i].setGravity(Gravity.CENTER);
            tv[i].setId(i+1);

        }
        //tv[0].setLayoutParams(tvsize);
        tvsize.addRule(RelativeLayout.RIGHT_OF, tv[0].getId());
        tv[1].setLayoutParams(tvsize);
        relative.addView(tv[0]);
        relative.addView(tv[1]);
        setContentView(relative);
    }

    public void text(){
        LinearLayout.LayoutParams orginal = new
                LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.MATCH_PARENT);
        // for the large one
        LinearLayout linear = new LinearLayout(this);
        linear.setLayoutParams(orginal);
        linear.setBackgroundColor(Color.parseColor("#FFFFFF"));
        linear.setOrientation(LinearLayout.VERTICAL);
        linear.setGravity(Gravity.CENTER);
        // three small one
        LinearLayout.LayoutParams horzon = new
                LinearLayout.LayoutParams(LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT);


        LinearLayout.LayoutParams textview_size = new
                LinearLayout.LayoutParams(200,200);
        textview_size.setMargins(10,10,10,10);


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
                tv.setLayoutParams(textview_size);
                tv.setBackgroundColor(Color.parseColor("#FF9800"));
                tv.setGravity(Gravity.CENTER);
                linear_line.addView(tv);
            }

            linear.addView(linear_line);
        }
        setContentView(linear);
    }
    public void color(){
        LinearLayout.LayoutParams linearpr = new
                LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.MATCH_PARENT); // default unit is px in layoutparams
        //linearpr.setMargins(10,10,10,10); // default unit is px

        LinearLayout.LayoutParams linearpr2 = new
                LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT,
                10); // default unit is px in layoutparams
        //linearpr.setMargins(10,10,10,10); // default unit is px

        LinearLayout linear = new LinearLayout(this);
        linear.setLayoutParams(linearpr);
        linear.setBackgroundColor(Color.parseColor("#000000"));
        linear.setOrientation(linear.VERTICAL);
        linear.setGravity(Gravity.CENTER);
        int id = 10; // only use int not byte, long...
        linear.setId(id);

        for(int i=0;i<255;i++){
            LinearLayout slinear = new LinearLayout(this);
            slinear.setLayoutParams(linearpr2);
            slinear.setBackgroundColor(Color.argb(255,i,0,0));
            linear.addView(slinear);
        }

        setContentView(linear);
    }
}