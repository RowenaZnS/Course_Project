import express from 'express';
import multer from 'multer';
import fs from 'fs/promises';
import { validate_user, update_user, fetch_user, username_exist} from './userdb.js';
var app = express();
let route = express.Router();
const form = multer();
const user_database = new Map();
route.get('/movie', async (req, res) => {
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