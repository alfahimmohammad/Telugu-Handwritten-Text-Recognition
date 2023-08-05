import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'dart:io';

class Home extends StatefulWidget {
  @override
  _HomeState createState() => _HomeState();
}

class _HomeState extends State<Home> {
  File image;

  getImageFromCamera() async {
    File _img = await ImagePicker.pickImage(source: ImageSource.camera);
    this.setState(() {
      image = _img;
    });

    Navigator.pushNamed(context, '/cropimage',arguments: {
      'img': image
    });
  }
  getImageFromGallery() async {
    File _img = await ImagePicker.pickImage(source: ImageSource.gallery);
    this.setState(() {
      image = _img;
    });

    Navigator.pushNamed(context, '/cropimage',arguments: {
      'img': image
    });
  }
  

  @override
  Widget build(BuildContext context) {
    return Scaffold(
    appBar: AppBar(
      title: Text('Telugu HTR'),
    ),
    body: Center(
    child:  Column(
      mainAxisAlignment: MainAxisAlignment.center,
        children: <Widget>[ 
          Container(
            child: Text('Make sure your phone and server are in the same LAN. Background processing might take longer if the text is large',
            style: TextStyle(fontSize: 20),
            ),
            margin: EdgeInsets.all(10),
          ),
          Container(child: RaisedButton(
            onPressed: () => getImageFromCamera(),
            child: Text('Snap image from camera',
              style: TextStyle(
                fontSize: 20,
                ),
              ),
            ),
            margin: EdgeInsets.all(10), 
          ),
          Container(child: RaisedButton(
            onPressed: () => getImageFromGallery(),
            child: Text('Import image from gallery',
              style: TextStyle(
                fontSize: 20,
                ),
              ),
            ),
            margin: EdgeInsets.all(10),
          ),
        ],  
    ),
   ),
);
}
}