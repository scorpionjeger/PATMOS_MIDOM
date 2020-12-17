// This faddapted from the file FromMVG.cpp

// with the following copyrite information

// Copyright (c) 2016
// cDc <cdc.seacave@gmail.com>
// Pierre MOULON

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.




#include "Interface.h"
#define _USE_EIGEN





#include <stdlib.h>
#include <stdio.h>
#include <cmath>
#include <iterator>
#include <iomanip>
#include <vector>
#include <set>
#include <iostream>
#include <jsoncpp/json/json.h>
#include <fstream>


bool exportToOpenMVS(

  const Json::Value & obj,
  const std::string & sOutFile,
  const std::string & sOutDir
  )
{


  
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  // Export data :
  MVS::Interface scene;



    const uint32_t nViews=10;

      MVS::Interface::Platform platform;
      // add the camera
      MVS::Interface::Platform::Camera camera;

      
      cv::Matx<double,3,3> Kk;

      Kk(0,0)=obj["CameraMatrix"]["0"][0].asDouble();
      Kk(1,0)=obj["CameraMatrix"]["0"][1].asDouble();
      Kk(2,0)=obj["CameraMatrix"]["0"][2].asDouble();

      Kk(0,1)=obj["CameraMatrix"]["1"][0].asDouble();
      Kk(1,1)=obj["CameraMatrix"]["1"][1].asDouble();
      Kk(2,1)=obj["CameraMatrix"]["1"][2].asDouble();

      Kk(0,2)=obj["CameraMatrix"]["2"][0].asDouble();
      Kk(1,2)=obj["CameraMatrix"]["2"][1].asDouble();
      Kk(2,2)=obj["CameraMatrix"]["2"][2].asDouble();  

     // width 1280 height 720
     
      camera.K = Kk;

      //const double fScale(1.0/std::max(imageHeader.width, imageHeader.height));
      const double fScale(1.0/std::max(1280, 720));
      camera.K(0, 0) *= fScale;
      camera.K(1, 1) *= fScale;
      camera.K(0, 2) *= fScale;
      camera.K(1, 2) *= fScale;

      camera.R = cv::Matx<double,3,3>::eye(); // Camera doesn't have any inherent rotation
      camera.C = cv::Point3_<double>(0,0,0); // or translation

      platform.cameras.push_back(camera);
      scene.platforms.push_back(platform);
//    }
//  }

  
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// define images & poses
  scene.images.reserve(obj["CamOrgn"].size());



    for (int itr=0;itr!=obj["CamOrgn"].size();itr++)
  {

    MVS::Interface::Image image;

    image.name=sOutDir+"/"+obj["Images"][std::to_string(itr)]["Imagename"].asString();

     std::cout<<obj["Images"][std::to_string(itr)]["Imagename"]<< "     image name thing\n";

    //image.platformID = map_intrinsic.at(view.second->id_intrinsic);
    image.platformID = 0;

    
    //what is this???
    MVS::Interface::Platform& platform = scene.platforms[image.platformID];

    


    image.cameraID = 0;

      MVS::Interface::Platform::Pose pose;

      image.poseID = itr;

      std::cout<<obj["CamOrgn"][std::to_string(itr)][0].asDouble()<< " thing\n";
      std::cout<<obj["CamOrgn"].size()<< " thing\n";



      cv::Matx<double,3,3> Rr;


      Rr(0,0)=obj["CamRotMat"][std::to_string(itr)]["X"][0].asDouble();
      Rr(0,1)=obj["CamRotMat"][std::to_string(itr)]["X"][1].asDouble();
      Rr(0,2)=obj["CamRotMat"][std::to_string(itr)]["X"][2].asDouble();

      Rr(1,0)=obj["CamRotMat"][std::to_string(itr)]["Y"][0].asDouble();
      Rr(1,1)=obj["CamRotMat"][std::to_string(itr)]["Y"][1].asDouble();
      Rr(1,2)=obj["CamRotMat"][std::to_string(itr)]["Y"][2].asDouble();

      Rr(2,0)=obj["CamRotMat"][std::to_string(itr)]["Z"][0].asDouble();
      Rr(2,1)=obj["CamRotMat"][std::to_string(itr)]["Z"][1].asDouble();
      Rr(2,2)=obj["CamRotMat"][std::to_string(itr)]["Z"][2].asDouble(); 

      pose.R = Rr; // 

      //pose.R = cv::Matx<double,3,3>::eye();
      pose.C = cv::Point3_<double>(obj["CamOrgn"][std::to_string(itr)][0].asDouble(),obj["CamOrgn"][std::to_string(itr)][1].asDouble(),obj["CamOrgn"][std::to_string(itr)][2].asDouble()); //


      platform.poses.push_back(pose);



    scene.images.push_back(image);
    //++my_progress_bar;
  }


  
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



  // define structure

  scene.vertices.reserve(obj["Vertex"].size());



    for (int itr=0;itr!=obj["Vertex"].size();itr++)
  {

    MVS::Interface::Vertex vert;
    MVS::Interface::Vertex::ViewArr& views = vert.views;

        

	for(int vitr=0;vitr!=obj["Vertex"][std::to_string(itr)]["CamList"].size();vitr++)
	{

        MVS::Interface::Vertex::View view;




        view.imageID = obj["Vertex"][std::to_string(itr)]["CamList"][vitr].asInt();
 

        view.confidence = 0;
        views.push_back(view);

    }

        //std::cout<<views.size()<< " view thing\n";



    vert.X = cv::Point3_<float>(obj["Vertex"][std::to_string(itr)]["Vect"][0].asFloat(),obj["Vertex"][std::to_string(itr)]["Vect"][1].asFloat(),obj["Vertex"][std::to_string(itr)]["Vect"][2].asFloat());
    scene.vertices.push_back(vert);
 }

 



  // write OpenMVS data
  if (!MVS::ARCHIVE::SerializeSave(scene, sOutFile))
    return false;

  std::cout
    << "Scene saved to OpenMVS interface format:\n"
    << "\t" << scene.images.size() << " images (" << "nPoses" << " calibrated)\n"
    << "\t" << scene.vertices.size() << " Landmarks\n"
    <<"\t"<<platform.poses.size()<< "poses\n";
  return true;
}


//from ORB_SLAM
int main(int argc, char **argv)

{
  //CmdLine cmd;
  std::string sSfM_Data_Filename="yes";
  std::string sOutFile = "scene.mvs";
  std::string sOutDir = "undistorted_images";



  std::ifstream ORBjson("data_file.json",std::ifstream::binary);

  Json::Reader reader;
  Json::Value obj;
  reader.parse(ORBjson,obj);

  if (exportToOpenMVS(obj, sOutFile, sOutDir))
    return( EXIT_SUCCESS );
  return( EXIT_FAILURE );
}

