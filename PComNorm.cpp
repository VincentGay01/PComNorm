#define GLM_ENABLE_EXPERIMENTAL
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/string_cast.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
std::vector<float> normals;
bool poseFound = false;
bool modelbool = false;
// Variables globales pour la caméra
float objectRotationX = 0.0f;
float objectRotationY = 0.0f;
float cameraDistance = 1.0f; // Distance de la caméra par rapport à l'objet
float cameraAngleX = 0.0f;   // Angle de rotation autour de l'axe X
float cameraAngleY = 0.0f;   // Angle de rotation autour de l'axe Y
bool isDragging = false;     // Indique si la souris est en train de glisser
double lastMouseX, lastMouseY;
glm::vec3 cameraTarget = glm::vec3(0.0f); // Cible de la caméra (peut être déplacée)
float cameraSpeed = 0.001f;               // Vitesse de déplacement de la caméra
bool keys[1024] = { false };            // État des touches
glm::mat4 alignmentMatrix(1.0f); // Matrice d'alignement pour le plan d'overlay
std::vector<float> vertices; // Variables pour le maillage
std::vector<unsigned int> indices; // Indices du maillage
// Variables pour le plan d'overlay
GLuint overlayVAO, overlayVBO, overlayEBO;
GLuint overlayTexture;
GLuint overlayShaderProgram;
GLuint shaderProgram;
glm::vec3 cameraPos;
glm::mat4 view;
glm::mat4 newview;
float overlayZ = -0.5f; // Distance entre la caméra et le plan d'overlay 


// Matrices de vue et de projection
glm::mat4 projection = glm::perspective(glm::radians(60.f), 1.0f, 0.1f, 1000.0f);
glm::mat4 Model;
glm::mat4 MVP;
// Callback pour ajuster la taille de la fenêtre
void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}


// Callback pour gérer les entrées du clavier
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode) {
    if (key >= 0 && key < 1024) {
        if (action == GLFW_PRESS) {
            keys[key] = true;
        }
        else if (action == GLFW_RELEASE) {
            keys[key] = false;
        }
    }
}





// Callback pour gérer les entrées de la souris
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            isDragging = true;
            glfwGetCursorPos(window, &lastMouseX, &lastMouseY);
        }
        else if (action == GLFW_RELEASE) {
            isDragging = false;
        }
    }
}

// Callback pour gérer le mouvement de la souris
void cursor_position_callback(GLFWwindow* window, double xpos, double ypos) {
    if (isDragging) {
        double dx = xpos - lastMouseX;
        double dy = ypos - lastMouseY;
        lastMouseX = xpos;
        lastMouseY = ypos;

        // Ajuster les angles de rotation de l'objet
        objectRotationX += static_cast<float>(dy) * 0.1f;
        objectRotationY += static_cast<float>(dx) * 0.1f;
    }
}

// Callback pour gérer le zoom avec la molette
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    cameraDistance -= static_cast<float>(yoffset) * 0.05f;
    if (cameraDistance < 0.001f) cameraDistance = 0.001f; // Limiter le zoom
    if (cameraDistance > 20.0f) cameraDistance = 20.0f; // Limiter le dézoom
}

// Fonction pour compiler un shader
GLuint compileShader(GLenum type, const char* source) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    // Vérification des erreurs de compilation
    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, nullptr, infoLog);
        std::cerr << "Erreur de compilation du shader : " << infoLog << std::endl;
    }
    return shader;
}

// Fonction pour créer un programme shader
GLuint createShaderProgram(const char* vertexSource, const char* fragmentSource) {
    GLuint vertexShader = compileShader(GL_VERTEX_SHADER, vertexSource);
    GLuint fragmentShader = compileShader(GL_FRAGMENT_SHADER, fragmentSource);

    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    // Vérification des erreurs de liaison
    GLint success;
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetProgramInfoLog(shaderProgram, 512, nullptr, infoLog);
        std::cerr << "Erreur de liaison du programme shader : " << infoLog << std::endl;
    }

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    return shaderProgram;
}

// Fonction pour lire un fichier .ply avec support des normales
bool loadPLY(const std::string& filename, std::vector<float>& vertices, std::vector<unsigned int>& indices, std::vector<float>& normals) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Erreur : Impossible d'ouvrir le fichier " << filename << std::endl;
        return false;
    }

    std::string line;
    bool headerEnded = false;
    int vertexCount = 0, faceCount = 0;
    bool hasNormals = false;

    // Lecture de l'en-tête
    while (std::getline(file, line)) {
        if (line == "end_header") {
            headerEnded = true;
            break;
        }
        std::istringstream iss(line);
        std::string word;
        iss >> word;
        if (word == "element") {
            std::string type;
            int count;
            iss >> type >> count;
            if (type == "vertex") vertexCount = count;
            if (type == "face") faceCount = count;
        }
        else if (word == "property" && line.find("nx") != std::string::npos) {
            hasNormals = true;
        }
    }

    if (!headerEnded) {
        std::cerr << "Erreur : En-tête du fichier .ply incorrect" << std::endl;
        return false;
    }

    // Variables temporaires pour stocker les positions
    std::vector<glm::vec3> positions;

    // Lecture des sommets
    for (int i = 0; i < vertexCount; ++i) {
        std::getline(file, line);
        std::istringstream iss(line);
        float x, y, z;
        iss >> x >> y >> z;

        // Stocker la position pour le calcul des normales si nécessaire
        positions.push_back(glm::vec3(x, y, z));

        // Ajouter la position aux vertices
        vertices.push_back(x);
        vertices.push_back(y);
        vertices.push_back(z);

        if (hasNormals) {
            // Si le fichier contient des normales, les lire
            float nx, ny, nz;
            iss >> nx >> ny >> nz;
            normals.push_back(nx);
            normals.push_back(ny);
            normals.push_back(nz);
        }
    }

    // Stocker les triangles pour calculer les normales si nécessaire
    std::vector<glm::uvec3> triangles;

    // Lecture des faces
    for (int i = 0; i < faceCount; ++i) {
        std::getline(file, line);
        std::istringstream iss(line);
        int vertexCount, v1, v2, v3;
        iss >> vertexCount >> v1 >> v2 >> v3;
        if (vertexCount == 3) { // On ne gère que les triangles
            indices.push_back(v1);
            indices.push_back(v2);
            indices.push_back(v3);

            // Stocker le triangle pour le calcul des normales
            triangles.push_back(glm::uvec3(v1, v2, v3));
        }
    }

    // Si le fichier ne contient pas de normales, les calculer
    if (!hasNormals) {
        // Initialiser les normales à zéro
        normals.resize(positions.size() * 3, 0.0f);

        // Calculer les normales pour chaque triangle
        for (const auto& triangle : triangles) {
            glm::vec3 v0 = positions[triangle[0]];
            glm::vec3 v1 = positions[triangle[1]];
            glm::vec3 v2 = positions[triangle[2]];

            // Calculer les vecteurs des côtés du triangle
            glm::vec3 edge1 = v1 - v0;
            glm::vec3 edge2 = v2 - v0;

            // Calculer la normale du triangle par produit vectoriel
            glm::vec3 normal = glm::normalize(glm::cross(edge1, edge2));

            // Ajouter cette normale aux trois sommets du triangle
            for (int j = 0; j < 3; ++j) {
                int idx = triangle[j] * 3;
                normals[idx] += normal.x;
                normals[idx + 1] += normal.y;
                normals[idx + 2] += normal.z;
            }
        }

        // Normaliser toutes les normales
        for (size_t i = 0; i < normals.size(); i += 3) {
            glm::vec3 n(normals[i], normals[i + 1], normals[i + 2]);
            n = glm::normalize(n);
            normals[i] = n.x;
            normals[i + 1] = n.y;
            normals[i + 2] = n.z;
        }
    }

    return true;
}

// Fonction pour projeter les sommets sur l'image
std::vector<cv::Point2f> projectVerticesToImage(
    const std::vector<float>& vertices,
    const glm::mat4& MVP,
    int image_width,
    int image_height)
{
    std::vector<cv::Point2f> projectedPoints;

    for (size_t i = 0; i < vertices.size(); i += 3) {
        glm::vec4 vertex(vertices[i], vertices[i + 1], vertices[i + 2], 1.0f);
        glm::vec4 clipSpace = MVP * vertex;

        if (clipSpace.w == 0.0f) continue; // éviter division par zéro

        glm::vec3 ndc = glm::vec3(clipSpace) / clipSpace.w; // Normalized Device Coordinates [-1,1]

        float x_img = (ndc.x * 0.5f + 0.5f) * image_width;
        float y_img = (1.0f - (ndc.y * 0.5f + 0.5f)) * image_height; // inverser Y pour OpenCV

        projectedPoints.push_back(cv::Point2f(x_img, y_img));
    }

    return projectedPoints;
}

std::vector<std::pair<cv::Point2f, glm::vec3>> matchKeypointsTo3D(
    const std::vector<cv::KeyPoint>& keypoints,
    const std::vector<cv::Point2f>& projectedVertices,
    const std::vector<float>& vertices3D,
    float maxDistancePixels = 3.0f) // Réduire le seuil pour plus de précision
{
    std::vector<std::pair<cv::Point2f, glm::vec3>> matches;

    if (projectedVertices.empty() || vertices3D.empty()) return matches;

    // Construire la matrice de recherche pour FLANN
    cv::Mat projectedMat(projectedVertices.size(), 2, CV_32F);
    for (size_t i = 0; i < projectedVertices.size(); ++i) {
        projectedMat.at<float>(i, 0) = projectedVertices[i].x;
        projectedMat.at<float>(i, 1) = projectedVertices[i].y;
    }

    // Création du KD-Tree FLANN
    cv::flann::KDTreeIndexParams indexParams(5); // 5 arbres
    projectedMat.convertTo(projectedMat, CV_32F);
    cv::flann::Index kdtree(projectedMat, indexParams);

    for (const auto& kp : keypoints) {
        cv::Mat query(1, 2, CV_32F);
        query.at<float>(0) = kp.pt.x;
        query.at<float>(1) = kp.pt.y;

        std::vector<int> indices(1);
        std::vector<float> dists(1);

        kdtree.knnSearch(query, indices, dists, 1, cv::flann::SearchParams(32));

        if (!indices.empty() && dists[0] < maxDistancePixels * maxDistancePixels) {
            int idx = indices[0];

            glm::vec3 corresponding3D(
                vertices3D[3 * idx],
                vertices3D[3 * idx + 1],
                vertices3D[3 * idx + 2]
            );

            matches.emplace_back(kp.pt, corresponding3D);
        }
    }

    // Ajouter un filtre supplémentaire pour rejeter les correspondances douteuses
    std::vector<std::pair<cv::Point2f, glm::vec3>> filteredMatches;
    for (const auto& match : matches) {
        // Vous pourriez ajouter des vérifications supplémentaires ici
        // Par exemple, vérifier si la profondeur Z est raisonnable
        if (match.second.z > -10.0f && match.second.z < 10.0f) {
            filteredMatches.push_back(match);
        }
    }

    return filteredMatches;
}


// fonction Pnp pour estimer pose de la cam
bool estimateCameraPoseFromMatches(
    const std::vector<std::pair<cv::Point2f, glm::vec3>>& matches,
    const cv::Mat& cameraMatrix,
    cv::Mat& rvec,
    cv::Mat& tvec,
    bool useExtrinsicGuess = false)
{
    if (matches.size() < 4) {
        std::cerr << "Pas assez de correspondances pour solvePnP (minimum 4 requis)" << std::endl;
        return false;
    }

    // Convertir les correspondances en vecteurs pour solvePnP
    std::vector<cv::Point3f> objectPoints;
    std::vector<cv::Point2f> imagePoints;

    for (const auto& match : matches) {
        imagePoints.push_back(match.first); // 2D
        objectPoints.emplace_back(
            match.second.x,
            match.second.y,
            match.second.z
        ); // 3D
    }

    // Aucune distorsion supposée (ou tu peux la passer en paramètre si tu veux)
    cv::Mat distCoeffs = cv::Mat::zeros(5, 1, CV_64F);

    bool success = cv::solvePnPRansac(
        objectPoints,
        imagePoints,
        cameraMatrix,
        distCoeffs,
        rvec,
        tvec,
        useExtrinsicGuess,
        50000,           // Augmenter le nombre d'itérations
        7.0,            // Seuil de reprojection
        0.99,           // Niveau de confiance
        cv::noArray(),  // Inliers
        cv::SOLVEPNP_EPNP  // Utiliser EPNP au lieu d'ITERATIVE
    );

    // Raffiner la pose si la résolution initiale a réussi
    if (success) {
        cv::solvePnPRefineLM(objectPoints, imagePoints, cameraMatrix,
            distCoeffs, rvec, tvec);
    }

    return success;
}

glm::mat4 createViewMatrixFromPnP(const cv::Mat& rvec, const cv::Mat& tvec)
{
    // Convertir le vecteur de rotation en matrice de rotation
    cv::Mat R;
    cv::Rodrigues(rvec, R);

    // Créer une matrice de transformation OpenGL
    glm::mat4 viewMatrix = glm::mat4(1.0f);

    // Remplir la matrice de rotation (transposée pour OpenGL)
    // OpenCV: X droite, Y bas, Z avant
    // OpenGL: X droite, Y haut, Z arrière
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            viewMatrix[i][j] = R.at<double>(j, i);
        }
    }

    // Inverser Y et Z pour la conversion OpenCV -> OpenGL
    viewMatrix[1][0] *= -1; viewMatrix[1][1] *= -1; viewMatrix[1][2] *= -1;
    viewMatrix[2][0] *= -1; viewMatrix[2][1] *= -1; viewMatrix[2][2] *= -1;

    // Appliquer la translation
    viewMatrix[3][0] = tvec.at<double>(0);
    viewMatrix[3][1] = -tvec.at<double>(1); // Inverser Y
    viewMatrix[3][2] = -tvec.at<double>(2); // Inverser Z

    // En OpenGL, nous voulons l'inverse de cette matrice pour la vue
    return glm::inverse(viewMatrix);
}


//fonction pour prendre photo de la scène
cv::Mat renderMeshToImage(const std::vector<float>& vertices,
    const std::vector<unsigned int>& indices,
    const std::vector<float>& normals, // Ajout des normales en paramètre
    GLuint shaderProgram,
    const glm::mat4& view,
    const glm::mat4& projection,
    const glm::mat4& model,
    int width = 2704,
    int height = 1800) {

    // Sauvegarder l'état actuel de OpenGL
    GLint previousFBO;
    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &previousFBO);
    GLint previousViewport[4];
    glGetIntegerv(GL_VIEWPORT, previousViewport);

    // Créer un FBO (Frame Buffer Object)
    GLuint fbo, renderTexture, depthBuffer;

    // Initialiser le FBO
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);

    // Créer une texture pour stocker le rendu
    glGenTextures(1, &renderTexture);
    glBindTexture(GL_TEXTURE_2D, renderTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, renderTexture, 0);

    // Créer un renderbuffer pour la profondeur
    glGenRenderbuffers(1, &depthBuffer);
    glBindRenderbuffer(GL_RENDERBUFFER, depthBuffer);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthBuffer);

    // Vérifier si le FBO est complet
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        std::cerr << "Erreur lors de la création du FBO" << std::endl;

        // Nettoyer les ressources en cas d'erreur
        glBindFramebuffer(GL_FRAMEBUFFER, previousFBO);
        glViewport(previousViewport[0], previousViewport[1], previousViewport[2], previousViewport[3]);
        glDeleteTextures(1, &renderTexture);
        glDeleteRenderbuffers(1, &depthBuffer);
        glDeleteFramebuffers(1, &fbo);

        return cv::Mat();
    }

    // Configurer la vue de rendu
    glViewport(0, 0, width, height);
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f); // Même couleur d'arrière-plan que l'écran principal
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Activer le test de profondeur si ce n'est pas déjà fait
    GLboolean depthTestWasEnabled;
    glGetBooleanv(GL_DEPTH_TEST, &depthTestWasEnabled);
    if (!depthTestWasEnabled) {
        glEnable(GL_DEPTH_TEST);
    }

    // Créer un VAO et VBO temporaires pour ce rendu
    GLuint tempVAO, tempVBO, tempNormalVBO, tempEBO;
    glGenVertexArrays(1, &tempVAO);
    glGenBuffers(1, &tempVBO);
    glGenBuffers(1, &tempNormalVBO); // Nouveau buffer pour les normales
    glGenBuffers(1, &tempEBO);

    glBindVertexArray(tempVAO);

    // Buffer pour les positions
    glBindBuffer(GL_ARRAY_BUFFER, tempVBO);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Buffer pour les normales
    glBindBuffer(GL_ARRAY_BUFFER, tempNormalVBO);
    glBufferData(GL_ARRAY_BUFFER, normals.size() * sizeof(float), normals.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);

    // Buffer pour les indices
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, tempEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);

    // Utiliser le shader et configurer ses uniformes
    glUseProgram(shaderProgram);

    // Utiliser EXACTEMENT le même modèle et matrices que dans l'affichage principal
    glm::mat4 MVP = projection * view * model;
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "MVP"), 1, GL_FALSE, glm::value_ptr(MVP));
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));

    // Définir les mêmes paramètres d'éclairage que ceux utilisés pour l'affichage principal
    glm::vec3 lightPos = glm::vec3(2.0f, 2.0f, 2.0f);
    glm::vec3 lightColor = glm::vec3(1.0f, 1.0f, 1.0f);
    glm::vec3 objectColor = glm::vec3(0.7f, 0.7f, 0.7f);
    float ambientStrength = 0.3f;
    float specularStrength = 0.5f;
    float shininess = 32.0f;

    glUniform3fv(glGetUniformLocation(shaderProgram, "lightPos"), 1, glm::value_ptr(lightPos));
    glUniform3fv(glGetUniformLocation(shaderProgram, "viewPos"), 1, glm::value_ptr(cameraPos)); // Position globale de la caméra
    glUniform3fv(glGetUniformLocation(shaderProgram, "lightColor"), 1, glm::value_ptr(lightColor));
    glUniform3fv(glGetUniformLocation(shaderProgram, "objectColor"), 1, glm::value_ptr(objectColor));
    glUniform1f(glGetUniformLocation(shaderProgram, "ambientStrength"), ambientStrength);
    glUniform1f(glGetUniformLocation(shaderProgram, "specularStrength"), specularStrength);
    glUniform1f(glGetUniformLocation(shaderProgram, "shininess"), shininess);

    // Dessiner le maillage
    glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);

    // Lire les pixels du framebuffer
    std::vector<unsigned char> pixels(width * height * 3);
    glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());

    // Convertir en image OpenCV
    cv::Mat result = cv::Mat(height, width, CV_8UC3);

    // OpenGL rend du bas vers le haut, OpenCV stocke du haut vers le bas
    // Donc nous devons inverser verticalement l'image
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int glIndex = (y * width + x) * 3;
            // Inverser l'axe Y et convertir RGB à BGR (OpenCV utilise BGR)
            result.at<cv::Vec3b>(height - 1 - y, x)[0] = pixels[glIndex + 2]; // B
            result.at<cv::Vec3b>(height - 1 - y, x)[1] = pixels[glIndex + 1]; // G
            result.at<cv::Vec3b>(height - 1 - y, x)[2] = pixels[glIndex + 0]; // R
        }
    }

    // Nettoyer les ressources
    glDeleteVertexArrays(1, &tempVAO);
    glDeleteBuffers(1, &tempVBO);
    glDeleteBuffers(1, &tempNormalVBO);
    glDeleteBuffers(1, &tempEBO);

    glDeleteTextures(1, &renderTexture);
    glDeleteRenderbuffers(1, &depthBuffer);
    glDeleteFramebuffers(1, &fbo);

    // Restaurer l'état précédent d'OpenGL
    glBindFramebuffer(GL_FRAMEBUFFER, previousFBO);
    glViewport(previousViewport[0], previousViewport[1], previousViewport[2], previousViewport[3]);

    if (!depthTestWasEnabled) {
        glDisable(GL_DEPTH_TEST);
    }

    return result;
}


// Fonction pour convertir une homographie en matrice de modèle
glm::mat4 homographyToModelMatrix(const cv::Mat& H) {
    // Vérifier si l'homographie est valide
    if (H.empty() || H.rows != 3 || H.cols != 3) {
        std::cerr << "Homographie invalide" << std::endl;
        return glm::mat4(1.0f); // Retourner une matrice identité en cas d'erreur
    }

    // Décomposer l'homographie en rotation et translation
    // Nous avons besoin d'une matrice de caméra approchée
    cv::Mat cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
    cameraMatrix.at<double>(0, 0) = 2252.308217738484; // Valeur approximative de fx (focale x)
    cameraMatrix.at<double>(1, 1) = 2252.308217738484; // Valeur approximative de fy (focale y)
    cameraMatrix.at<double>(0, 2) = 1352.0; // Valeur approximative de cx (centre x)
    cameraMatrix.at<double>(1, 2) = 900.0; // Valeur approximative de cy (centre y)

    std::vector<cv::Mat> rotations, translations, normals;

    try {
        // Décomposer l'homographie en solutions possibles
        cv::decomposeHomographyMat(H, cameraMatrix, rotations, translations, normals);

        if (rotations.empty() || translations.empty()) {
            std::cerr << "Échec de la décomposition de l'homographie" << std::endl;
            return glm::mat4(1.0f);
        }

        std::cout << "Nombre de solutions possibles: " << rotations.size() << std::endl;

        // Sélectionner la première solution (index 0)
        // Dans une implémentation plus robuste, vous pourriez vouloir choisir
        // la meilleure solution selon un critère spécifique
        cv::Mat R = rotations[0];  // Matrice de rotation 3x3
        cv::Mat t = translations[0]; // Vecteur de translation 3x1

        // Convertir en matrice de modèle glm 4x4
        glm::mat4 modelMatrix(1.0f);

        // Remplir la sous-matrice 3x3 avec la rotation (transposée pour OpenGL)
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                // Transposer la matrice car OpenGL utilise une convention différente
                modelMatrix[j][i] = static_cast<float>(R.at<double>(i, j));
            }
        }

        // Ajouter la translation (ajuster selon les besoins)
        modelMatrix[3][0] = static_cast<float>(t.at<double>(0, 0));
        modelMatrix[3][1] = static_cast<float>(t.at<double>(1, 0));
        modelMatrix[3][2] = static_cast<float>(t.at<double>(2, 0));

        // Ajustements possibles pour aligner correctement
        // Ces valeurs peuvent nécessiter des ajustements selon votre setup
        float scaleFactor = 0.1f;  // Facteur d'échelle à ajuster selon les besoins

        glm::mat4 scaleMatrix = glm::scale(glm::mat4(1.0f), glm::vec3(scaleFactor));
        modelMatrix = modelMatrix * scaleMatrix;

        return modelMatrix;

    }
    catch (const cv::Exception& e) {
        std::cerr << "Exception pendant la décomposition de l'homographie: " << e.what() << std::endl;
        return glm::mat4(1.0f);
    }
}

// Fonction modifiée pour prendre en compte les matrices de vue et de projection actuelles
bool alignMeshWithImage(const char* imagePath,
    const std::vector<float>& vertices,
    const std::vector<unsigned int>& indices,
    const std::vector<float>& normals, // Ajout des normales
    GLuint shaderProgram,
     glm::mat4& view,
    const glm::mat4& projection,
    const glm::mat4& model,
    glm::mat4& outModelMatrix) {

    // 1. Charger l'image de référence
    cv::Mat referenceImage = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (referenceImage.empty()) {
        std::cerr << "Erreur: Impossible de charger l'image de référence: " << imagePath << std::endl;
        return false;
    }

    // 2. Préparer une image de rendu du maillage 3D avec la vue de caméra actuelle
    cv::Mat meshRendering = renderMeshToImage(vertices, indices, normals,
        shaderProgram, view, projection, model,
        referenceImage.cols, referenceImage.rows);

    // Optionnel: Sauvegarder les images pour débogage
    cv::imwrite("reference_image.png", referenceImage);
    cv::imwrite("mesh_rendering.png", meshRendering);

    // 3. Détecter des points caractéristiques dans les deux images
    cv::Ptr<cv::Feature2D> detector = cv::SIFT::create();
    std::vector<cv::KeyPoint> keypointsRef, keypointsMesh;
    cv::Mat descriptorsRef, descriptorsMesh;
    
    detector->detectAndCompute(referenceImage, cv::noArray(), keypointsRef, descriptorsRef);
    detector->detectAndCompute(meshRendering, cv::noArray(), keypointsMesh, descriptorsMesh);

    // Vérifier si des points caractéristiques ont été trouvés
    if (keypointsRef.empty() || keypointsMesh.empty()) {
        std::cerr << "Erreur: Aucun point caractéristique détecté dans les images" << std::endl;
        return false;
    }

    if (descriptorsRef.empty() || descriptorsMesh.empty()) {
        std::cerr << "Erreur: Pas de descripteurs générés pour les points caractéristiques" << std::endl;
        return false;
    }

    std::cout << "Points caractéristiques détectés: " << keypointsRef.size()
        << " dans l'image de référence, " << keypointsMesh.size()
        << " dans le rendu du maillage." << std::endl;

    // 4. Matcher les points caractéristiques
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    std::vector<std::vector<cv::DMatch>> knnMatches;

    // Vérifier si les descripteurs sont du bon type pour FLANN
    if (descriptorsRef.type() != CV_32F) {
        descriptorsRef.convertTo(descriptorsRef, CV_32F);
    }
    if (descriptorsMesh.type() != CV_32F) {
        descriptorsMesh.convertTo(descriptorsMesh, CV_32F);
    }

    try {
        matcher->knnMatch(descriptorsRef, descriptorsMesh, knnMatches, 2);
    }
    catch (const cv::Exception& e) {
        std::cerr << "Erreur lors du matching: " << e.what() << std::endl;
        return false;
    }

    // 5. Filtrer les correspondances de bonne qualité
    const float ratioThresh = 0.7f;
    std::vector<cv::DMatch> goodMatches;
    for (size_t i = 0; i < knnMatches.size(); i++) {
        if (knnMatches[i].size() < 2) continue; // Ignorer les matchs incomplets

        if (knnMatches[i][0].distance < ratioThresh * knnMatches[i][1].distance) {
            goodMatches.push_back(knnMatches[i][0]);
        }
    }

    std::cout << "Nombre de bonnes correspondances trouvées: " << goodMatches.size() << std::endl;

    if (goodMatches.size() < 8) {
        std::cerr << "Pas assez de bonnes correspondances trouvées (minimum 8 requises)" << std::endl;
        return false;
    }

	// fonction pour le calcul de la matrice MVP
    int image_width = referenceImage.cols;
    int image_height = referenceImage.rows;

    // 1. Projetter les sommets sur l'image
    glm::mat4 tempMVP = projection * view * model;
    std::vector<cv::Point2f> projectedVertices = projectVerticesToImage(vertices, tempMVP, image_width, image_height);

    // 2. Matcher les keypoints détectés sur l'image du mesh
    std::vector<std::pair<cv::Point2f, glm::vec3>> matchedPoints = matchKeypointsTo3D(keypointsMesh, projectedVertices, vertices);

    // 3. Utiliser les correspondances
    cv::Mat debugImage = meshRendering.clone();
    for (const auto& m : matchedPoints) {
        cv::circle(debugImage, m.first, 4, cv::Scalar(0, 255, 0), -1);
    }
    cv::imwrite("debug_matched_points.png", debugImage);
    for (const auto& match : matchedPoints) {
        const cv::Point2f& keypoint2D = match.first;
        const glm::vec3& point3D = match.second;
    }

    //-----------------------------------------------------------------------------

    cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) <<
        2252.308217738484, 0, 1352.0,
        0, 2252.308217738484, 900.0,
        0, 0, 1);
    cv::Mat rvec, tvec;
    
    //  Appeler la fonction
    poseFound = estimateCameraPoseFromMatches(matchedPoints, cameraMatrix, rvec, tvec);

    if (poseFound) {
        std::cout << "Rotation (Rodrigues) : " << rvec.t() << std::endl;
        std::cout << "Translation : " << tvec.t() << std::endl;

        // Optionnel : convertir rvec en matrice de rotation
        cv::Mat R;
        cv::Rodrigues(rvec, R);
        std::cout << "Rotation matrix : " << R << std::endl;
        
            newview = createViewMatrixFromPnP(rvec, tvec);

    }

    //-------------------------------------------------------------------
	// 5. Afficher les correspondances
    cv::Mat imgMatches;
    cv::drawMatches(referenceImage, keypointsRef, meshRendering, keypointsMesh,
        goodMatches, imgMatches, cv::Scalar::all(-1),
        cv::Scalar::all(-1), std::vector<char>(),
        cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cv::imwrite("matches.png", imgMatches);

    // 6. Extraire les points correspondants
    std::vector<cv::Point2f> pointsRef, pointsMesh;
    for (size_t i = 0; i < goodMatches.size(); i++) {
        pointsRef.push_back(keypointsRef[goodMatches[i].queryIdx].pt);
        pointsMesh.push_back(keypointsMesh[goodMatches[i].trainIdx].pt);
    }

    // 7. Calculer l'homographie entre les deux ensembles de points
    cv::Mat H;
    try {
        H = cv::findHomography(pointsMesh, pointsRef, cv::RANSAC, 3.0);

        if (H.empty()) {
            std::cerr << "Erreur: Impossible de calculer l'homographie" << std::endl;
            return false;
        }
    }
    catch (const cv::Exception& e) {
        std::cerr << "Erreur lors du calcul de l'homographie: " << e.what() << std::endl;
        return false;
    }

    // 8. Convertir cette homographie en matrice de transformation 3D
    outModelMatrix = homographyToModelMatrix(H);

    std::cout << "Alignement réussi. Matrice de transformation générée." << std::endl;
    return true;
}


float overlayScale = 0.8f;


// Modification de la fonction processInput pour utiliser la vue de caméra actuelle
void processInput() {
    // Définir des vecteurs d'axes mondiaux
    glm::vec3 worldRight = glm::vec3(1.0f, 0.0f, 0.0f);   // Axe X
    glm::vec3 worldUp = glm::vec3(0.0f, 1.0f, 0.0f);      // Axe Y
    glm::vec3 worldForward = glm::vec3(0.0f, 0.0f, -1.0f); // Axe Z (négatif car OpenGL)
    if (!poseFound) {


        if (keys[GLFW_KEY_W] || keys[GLFW_KEY_Z])
            cameraTarget += worldForward * cameraSpeed;
        if (keys[GLFW_KEY_S])
            cameraTarget -= worldForward * cameraSpeed;
        if (keys[GLFW_KEY_A] || keys[GLFW_KEY_Q])
            cameraTarget -= worldRight * cameraSpeed;
        if (keys[GLFW_KEY_D])
            cameraTarget += worldRight * cameraSpeed;
        if (keys[GLFW_KEY_SPACE])
            cameraTarget += worldUp * cameraSpeed;
        if (keys[GLFW_KEY_LEFT_CONTROL] || keys[GLFW_KEY_RIGHT_CONTROL])
            cameraTarget -= worldUp * cameraSpeed;
    }
    if (keys[GLFW_KEY_E]) {
        
        bool alignmentSuccess = alignMeshWithImage(
            "D:/project/PComNorm/baseColor.png", // Chemin de l'image de référence
            vertices,                            // Vos vertices
            indices,                             // Vos indices
            normals,                             // Vos normales
            shaderProgram,                       // Votre programme de shader
            view,                                // Vue actuelle de la caméra
            projection,                          // Projection actuelle
            Model,                               // Matrice modèle actuelle
            alignmentMatrix                      // Matrice de sortie
        );
    }

    // Déplacement haut/bas sur l'axe Y
    if (keys[GLFW_KEY_SPACE])
        cameraTarget += worldUp * cameraSpeed;
    if (keys[GLFW_KEY_LEFT_CONTROL] || keys[GLFW_KEY_RIGHT_CONTROL])
        cameraTarget -= worldUp * cameraSpeed;

    // Réinitialiser la position de la caméra
    if (keys[GLFW_KEY_R]) {
        cameraTarget = glm::vec3(0.0f);
        cameraAngleX = 0.0f;
        cameraAngleY = 0.0f;
        cameraDistance = 5.0f;
    }
}


int main() {
    // Initialisation de GLFW
    if (!glfwInit()) {
        std::cerr << "Échec de l'initialisation de GLFW" << std::endl;
        return -1;
    }

    // Configuration de la version OpenGL
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Création de la fenêtre
    GLFWwindow* window = glfwCreateWindow(2000,2000, "Fenêtre OpenGL", nullptr, nullptr);
    if (!window) {
        std::cerr << "Échec de la création de la fenêtre GLFW" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    // Initialisation de GLEW
    if (glewInit() != GLEW_OK) {
        std::cerr << "Échec de l'initialisation de GLEW" << std::endl;
        return -1;
    }

    // Définir la fonction de callback pour le redimensionnement
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback);
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetKeyCallback(window, key_callback);
    // Activer le blending pour la transparence
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Activer le test de profondeur
    glEnable(GL_DEPTH_TEST);

    // Configurer l'overlay
  //  setupOverlay();

    // Dans la fonction main(), ajouter ces variables pour l'éclairage
  // Pour stocker les normales du maillage

    // Charger le maillage avec les normales
    if (!loadPLY("D:/project/PComNorm/Chemi-AU-O005.ply", vertices, indices, normals)) {
        return -1;
    }

    // Création du VAO, VBO et EBO avec support pour les normales
    GLuint VAO, VBO, normalVBO, EBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &normalVBO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(VAO);

    // Buffer pour les positions
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Buffer pour les normales
    glBindBuffer(GL_ARRAY_BUFFER, normalVBO);
    glBufferData(GL_ARRAY_BUFFER, normals.size() * sizeof(float), normals.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);

    // Buffer pour les indices
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    // Shaders pour le modèle 3D avec illumination
    const char* vertexShaderSource = R"(
    #version 330 core
    layout (location = 0) in vec3 aPos;
    layout (location = 1) in vec3 aNormal; // Ajout des normales
    
    uniform mat4 Model;
    uniform mat4 view;
    uniform mat4 projection;
    uniform mat4 MVP;
    
    out vec3 FragPos;
    out vec3 Normal;
    
    void main() {
        gl_Position = MVP * vec4(aPos, 1.0);
        FragPos = vec3(Model * vec4(aPos, 1.0));
        // Transposer l'inverse de la matrice modèle pour les normales
        Normal = mat3(transpose(inverse(Model))) * aNormal;
    }
)";

    const char* fragmentShaderSource = R"(
    #version 330 core
    in vec3 FragPos;
    in vec3 Normal;
    
    out vec4 FragColor;
    
    // Paramètres d'éclairage
    uniform vec3 lightPos;
    uniform vec3 viewPos;
    uniform vec3 lightColor;
    uniform vec3 objectColor;
    uniform float ambientStrength;
    uniform float specularStrength;
    uniform float shininess;
    
    void main() {
        // Composante ambiante
        vec3 ambient = ambientStrength * lightColor;
        
        // Composante diffuse
        vec3 norm = normalize(Normal);
        vec3 lightDir = normalize(lightPos - FragPos);
        float diff = max(dot(norm, lightDir), 0.0);
        vec3 diffuse = diff * lightColor;
        
        // Composante spéculaire
        vec3 viewDir = normalize(viewPos - FragPos);
        vec3 reflectDir = reflect(-lightDir, norm);
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
        vec3 specular = specularStrength * spec * lightColor;
        
        // Combinaison des composantes
        vec3 result = (ambient + diffuse + specular) * objectColor;
        FragColor = vec4(result, 1.0);
    }
)";

    shaderProgram = createShaderProgram(vertexShaderSource, fragmentShaderSource);

    // Paramètres d'éclairage par défaut
    glm::vec3 lightPos = glm::vec3(2.0f, 2.0f, 2.0f);
    glm::vec3 lightColor = glm::vec3(1.0f, 1.0f, 1.0f);
    glm::vec3 objectColor = glm::vec3(0.7f, 0.7f, 0.7f);
    float ambientStrength = 0.3f;
    float specularStrength = 0.5f;
    float shininess = 32.0f;

    // Boucle de rendu
    while (!glfwWindowShouldClose(window)) {
        // Effacer les buffers de couleur et de profondeur
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        processInput();
        if (poseFound == false) {
            cameraPos = glm::vec3(
                cameraDistance * sin(glm::radians(cameraAngleY)) * cos(glm::radians(cameraAngleX)),
                cameraDistance * sin(glm::radians(cameraAngleX)),
                cameraDistance * cos(glm::radians(cameraAngleY)) * cos(glm::radians(cameraAngleX))
            );

            cameraPos += cameraTarget;
           
        }
        else {
            glm::mat4 invView = glm::inverse(newview);
            cameraPos = glm::vec3(invView[3]);
			cameraPos = -cameraPos;
        }
        view = glm::lookAt(cameraPos, cameraTarget, glm::vec3(0.0f, 1.0f, 0.0f));
        
        
        if (modelbool==false)
        {
            // Matrice Model pour l'objet 3D
            Model = glm::mat4(1.0f);
            // Model = translate(Model, glm::vec3(0, 0, cameraDistance));
            Model = rotate(Model, glm::radians(objectRotationX), glm::vec3(1, 0, 0));
            Model = rotate(Model, glm::radians(objectRotationY), glm::vec3(0, 1, 0));
            Model = scale(Model, glm::vec3(.8, .8, .8));
           
        }

        MVP = projection * view * Model;

        // Dessiner d'abord l'objet 3D
        glUseProgram(shaderProgram);
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "MVP"), 1, GL_FALSE, &MVP[0][0]);
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "Model"), 1, GL_FALSE, &Model[0][0]);


        // Passer les paramètres d'éclairage au shader
        glUniform3fv(glGetUniformLocation(shaderProgram, "lightPos"), 1, glm::value_ptr(lightPos));
        glUniform3fv(glGetUniformLocation(shaderProgram, "viewPos"), 1, glm::value_ptr(cameraPos));
        glUniform3fv(glGetUniformLocation(shaderProgram, "lightColor"), 1, glm::value_ptr(lightColor));
        glUniform3fv(glGetUniformLocation(shaderProgram, "objectColor"), 1, glm::value_ptr(objectColor));
        glUniform1f(glGetUniformLocation(shaderProgram, "ambientStrength"), ambientStrength);
        glUniform1f(glGetUniformLocation(shaderProgram, "specularStrength"), specularStrength);
        glUniform1f(glGetUniformLocation(shaderProgram, "shininess"), shininess);

        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);
        // Échange des buffers
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Nettoyage
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);
    glDeleteProgram(shaderProgram);

    glDeleteVertexArrays(1, &overlayVAO);
    glDeleteBuffers(1, &overlayVBO);
    glDeleteBuffers(1, &overlayEBO);
    glDeleteBuffers(1, &normalVBO);
    glDeleteProgram(overlayShaderProgram);

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}