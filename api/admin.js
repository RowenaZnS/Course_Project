import express from 'express';
import multer from 'multer';
import path from 'path';
import fs from 'fs/promises';
import {fetch_movie,update_movie,fetch_orders} from './db_movielist.js';
var app = express();
let route = express.Router();
const form = multer();
const user_database = new Map();
route.get('/list', async (req, res) => {
    if (req.session.logged) {
      const movielist = await fetch_movie(); // Retrieve user from the in-memory database
      //console.log(movielist);
      return res.json({
        status: 'success',
        movielists: {
            movies:movielist
        },
      });
    }
    res.status(401).json({
      status: 'failed',
      message: 'Unauthorized',
    });
  });

// Configure disk storage
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, './public/media/moviedata/'); // Save to this directory
  },
  filename: function (req, file, cb) {
    // Extract the original file extension
    const ext = path.extname(file.originalname);
    // Generate a unique filename with the original extension
    const uniqueName = `${file.fieldname}-${Date.now()}${ext}`;
    cb(null, uniqueName);
  },
});
//   const fileFilter = function (req, file, cb) {
//     // 检查文件 MIME 类型
//     const allowedTypes = ['image/jpeg', 'image/png'];
  
//     if (allowedTypes.includes(file.mimetype)) {
//       cb(null, true); // 接受文件
//     } else {
//       cb(new Error('Only .jpg and .png files are allowed!'), false); // 拒绝文件
//     }
//   };
// const upload = multer({ dest: './public/media/moviedata/' });
const upload = multer({ storage: storage });

  route.post('/create', upload.any(), async (req, res) => {
    console.log(req.body);
    const {image, name, price, date, venue, introduction} = req.body; // Extract username and password
    console.log(req.body);
    if (name!= '' && price != '') {
      if (await update_movie(image,name, price, date, venue, introduction)) {
        return res.status(200).json({
          status: 'success',
          newmovie: {
            image:image,
            name: name,
            price: price,
            date: date,
            venue: venue,
            introduction: introduction,
          },
        });
      }
      else {
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

  route.get('/all_order', async (req, res) => {
    if (req.session.logged) {
      const user = await fetch_orders(); // Retrieve user from the in-memory database
      return res.json({
        status: 'success',
        user: {
          user
        },
      });
    }
  
    res.status(401).json({
      status: 'failed',
      message: 'Unauthorized',
    });
  });
  export{route};