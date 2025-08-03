package com.example.myapplication;

import android.graphics.Color;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.LinearLayout;

import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_main);
        Button bt1= (Button) findViewById(R.id.bt1);
        btclick btc = new btclick();
        bt1.setOnClickListener(btc);
        Button bt2= (Button) findViewById(R.id.bt2);
        btclick btc2 = new btclick();
        bt2.setOnClickListener(btc2);
    }
    class btclick implements View.OnClickListener{
        @Override
        public void onClick(View v){
            LinearLayout linear =(LinearLayout) findViewById(R.id.linear1);
            if(v.getId()==R.id.bt1){
                linear.setBackgroundColor(Color.WHITE);
            }else if(v.getId()==R.id.bt2){
                linear.setBackgroundColor(Color.GREEN);
            }
        }
    }
}