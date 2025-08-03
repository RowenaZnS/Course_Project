package com.example.p20250106;

import android.graphics.Color;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.view.Gravity;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup;
import android.widget.LinearLayout;
import android.widget.TableLayout;
import android.widget.TableRow;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

public class MainActivity extends AppCompatActivity {
    float counter = 0f;
    Handler handler = new Handler(Looper.getMainLooper());
    String num = "";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Getting the TableLayout and LinearLayout from XML
        TableLayout tl = findViewById(R.id.tl1);
        LinearLayout ll = findViewById(R.id.ll1);

        TableRow.LayoutParams tvSize = new TableRow.LayoutParams(200, 200);
        tvSize.setMargins(10, 10, 10, 10);

        ArrayList<Integer> b = new ArrayList<>();
        for (int i = 0; i < 7; i++) {
            b.add(i);
        }
        Collections.shuffle(b); // Randomize the order of numbers in b

        ArrayList<TableRow> rows = new ArrayList<>();
        ArrayList<TextView> tvList = new ArrayList<>();
        Random rand = new Random();

        for (int i = 0; i < 7; i++) {
            TableRow row = new TableRow(this);
            rows.add(row);

            int blackCellIndex = rand.nextInt(5); // Random position for black cell in row
            for (int j = 0; j < 5; j++) {
                TextView cell = new TextView(this);
                cell.setLayoutParams(tvSize);
                cell.setGravity(Gravity.CENTER);

                if (j == blackCellIndex) {
                    // Black cell
                    cell.setBackgroundColor(Color.BLACK);
                    cell.setText(String.valueOf(b.get(i) + 1));
                    cell.setTextColor(Color.WHITE);
                    cell.setId(View.generateViewId()); // Generate unique ID
                    cell.setOnTouchListener(new TTouch());
                    tvList.add(cell); // Add to tvList for future reference
                } else {
                    // White cell
                    cell.setBackgroundColor(Color.WHITE);
                }

                row.addView(cell);
            }
            tl.addView(row);
        }

        // Start a thread to update the counter and manage timeout
        Thread thr = new Thread(() -> {
            while (true) {
                counter++;
                handler.post(() -> {
                    if (counter > 5) {
                        for (TextView tv : tvList) {
                            tv.setBackgroundColor(Color.GRAY);
                            tv.setText("");
                        }
                    }
                });
                try {
                    Thread.sleep(1000);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
        });
        thr.start();
    }

    class TTouch implements View.OnTouchListener {
        @Override
        public boolean onTouch(View v, MotionEvent event) {
            if (event.getAction() == 0) {
                TextView tv = findViewById(R.id.tv1); // TextView to display the sequence
                String value = ((TextView) v).getText().toString();
                num += value;
                tv.setText(num);
                v.setVisibility(View.INVISIBLE); // Hide the touched view
            }
            return true;
        }
    }
}
