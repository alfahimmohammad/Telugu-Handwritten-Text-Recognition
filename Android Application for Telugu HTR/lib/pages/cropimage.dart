import 'package:flutter/material.dart';
import 'package:image_crop/image_crop.dart';

class Cropimage extends StatefulWidget {
  @override
  _CropimageState createState() => _CropimageState();
}

class _CropimageState extends State<Cropimage> {
  final cropKey = GlobalKey<CropState>();
  Map data = {};
  @override
  Widget build(BuildContext context) {
  data = ModalRoute.of(context).settings.arguments;
  var image = data['img'];
    return SafeArea(
        child: Container(
          //color: Colors.black,
          padding: const EdgeInsets.symmetric(vertical: 40.0, horizontal: 20.0),
          child: image == null ? Scaffold(
            body: Center(
              child: Text('No image selected',
                style: TextStyle(
                  fontSize: 20.0
                ),
              ),
            ),
          ): _buildCroppingImage(context, image),
        ),
      );
  }

  Widget _buildCroppingImage(context, var image) {
  return Column(
    children: <Widget>[
      Expanded(
        child: Crop.file(image, key: cropKey),
      ),
      Container(
        padding: const EdgeInsets.only(top: 20.0),
        alignment: AlignmentDirectional.center,
        child: Row(
          mainAxisAlignment: MainAxisAlignment.spaceAround,
          children: <Widget>[
            FlatButton(
              child: Text(
                'Crop Image',
                style: Theme.of(context)
                    .textTheme
                    .button
                    .copyWith(color: Colors.white),
              ),
              onPressed: () => _cropImage(context, image),
            ),
          ],
        ),
      )
    ],
  );
}

  Future<void> _cropImage(context, var image) async {
    //final scale = cropKey.currentState.scale;
    final crop = cropKey.currentState;
    final area = crop.area;
    if (area == null) {
      // cannot crop, widget is not setup
      return;
    }
    final options = await ImageCrop.getImageOptions(file: image);
    final width = options.width;
    final height = options.height;
    // scale up to use maximum possible number of pixels
    // this will sample image in higher resolution to make cropped image larger
    final sample = await ImageCrop.sampleImage(
      file: image,
      preferredWidth: width,
      preferredHeight: height,
    );
    final file = await ImageCrop.cropImage(
      file: sample,
      area: area,
    );
    Navigator.pushNamed(context, '/displayimage',arguments: {
      'img': file
    });
  }
}