import express from 'express';
import multer from 'multer';
import fs from 'fs/promises';
import { validate_user, update_user, fetch_user, username_exist} from './userdb.js';
import { updateorder} from './myorder.js'
var app = express();
let route = express.Router();
const form = multer();
const user_database = new Map();

route.post('/payment_page', form.none(), async (req, res) => {
  if (!req.session.logged) {
    return  res.status(401).json({
      status: 'failed',
      message: 'Login first',
    })
  }
  const {Event_id, movie_imagel, movie_name, time_select, seat_list,tickets,total_cost } = req.body; // Extract username and password 
  req.session.Event_id=Event_id;
  req.session.movie_imagel=movie_imagel;
  req.session.movie_name=movie_name;
  req.session.time_select=time_select;
  req.session.seat_list=seat_list;
  req.session.tickets=tickets;
  req.session.total_cost=total_cost;
    return res.json({
      status: 'success',
      payment_info: {

        time_select: time_select,
        seat_list: seat_list,
        tickets:tickets,
        total_cost:total_cost
      },
    });
  });

route.get('/payment_page', (req, res) => {// Retrieve user from the in-memory database
      return res.json({
        status: 'success',
        payment_info: {
          Event_id:req.session.Event_id,
          movie_imagel: req.session.movie_imagel,
          movie_name: req.session.movie_name,
          time_select: req.session.time_select,
          seat_list: req.session.seat_list,
          tickets:req.session.tickets,
          total_cost:req.session.total_cost
        },
      });
  });
// Define the request handler for POST /logout
route.post('/logout', (req, res) => {
  if (req.session.logged) {
    req.session.destroy((err) => {
      if (err) {
        return res.status(500).json({ status: 'failed', message: 'Unable to log out' });
      }
      res.end(); // Send an empty response
    });
  } else {
    res.status(401).json({
      status: 'failed',
      message: 'Unauthorized',
    });
  }
});

// Define the asynchronous request handler for GET /me
route.get('/me', async (req, res) => {
  if (req.session.logged) {
    const user = await fetch_user(req.session.username); // Retrieve user from the in-memory database
    return res.json({
      status: 'success',
      user: {
        username: user.username,
        role: user.role,
      },
    });
  }

  res.status(401).json({
    status: 'failed',
    message: 'Unauthorized',
  });
});
// Define the request handler for POST /register
route.post('/order', form.none(), async(req, res) => {
  const {firstName,lastName,email,address,region,district,paymentMethod, username,order_id,Purchase_Date,Event_id,Event_Name,Event_Date,seat_list,total_cost,tickets} = req.body; // Extract username and password
  if (username!="") {
    if(await updateorder(firstName,lastName,email,address,region,district,paymentMethod,username,order_id,Purchase_Date,Event_id,Event_Name,Event_Date,seat_list,total_cost,tickets)){
      return res.status(200).json({
        status: 'success',
        user: {
          order_id: order_id,
          Event_id: Event_id,
          Purchase_Date: Purchase_Date,
          Event_Name:Event_Name,
          Event_Date:Event_Date,
          seat_list:seat_list,
          total_cost:total_cost,
          tickets:tickets
        },
      });
    }
    else{
      return res.status(500).json({
        status: 'failed',
        message: 'Account created but unable to save into the database',
      });
    }
  } else {
    return res.status(400).json({
      status: 'failed',
      message: 'Missing fields',
    });
  }
});
export { route };
