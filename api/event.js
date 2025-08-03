import express from 'express';
import multer from 'multer';
import fs from 'fs/promises';
import client from './dbclient.js';
import { validate_user, update_user, fetch_user, username_exist} from './userdb.js';
var app = express();
let route = express.Router();
const form = multer();
var id;
route.post('/movie',form.none(), async (req, res) => {
  if (req.session.logged) {
    const { id } = req.body; // Retrieve user from the in-memory database
    req.session.movieId = id; 
    
    return res.json({
    status: 'success',
    movies: {
        id: id,
    },
    });
  }
  else {
    return res.status(401).json({
      status: 'failed',
      message: 'Please Login First!',
    });
  }
  });
  route.get('/movie', (req, res) => {
    id=req.session.event_id; // Retrieve user from the in-memory database
      return res.json({
        status: 'success',
        movies: {
            id: req.session.movieId,
        },
      });
  });
  export {route};
