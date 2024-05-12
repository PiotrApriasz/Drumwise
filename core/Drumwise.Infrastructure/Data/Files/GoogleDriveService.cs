using System.Reflection;
using Drumwise.Application.Common.Interfaces;
using Drumwise.Application.Common.Models.Settings;
using Drumwise.Application.Services.Files;
using Google.Apis.Auth.OAuth2;
using Google.Apis.Auth.OAuth2.Flows;
using Google.Apis.Auth.OAuth2.Responses;
using Google.Apis.Download;
using Google.Apis.Drive.v3;
using Google.Apis.Services;
using Google.Apis.Upload;
using Google.Apis.Util.Store;
using NLog;
using NLog.Fluent;

namespace Drumwise.Infrastructure.Data.Files;

public class GoogleDriveService(GoogleDriveApiSettings googleDriveApiSettings) : IFileStorageService
{
    
    private static readonly Logger Logger = LogManager.GetCurrentClassLogger();
    
    private DriveService AuthenticateGoogleDrive()
    {
        var tokenResponse = new TokenResponse()
        {
            AccessToken = googleDriveApiSettings.AccessToken,
            RefreshToken = googleDriveApiSettings.RefreshToken
        };

        var apiCodeFlow = new GoogleAuthorizationCodeFlow(new GoogleAuthorizationCodeFlow.Initializer()
        {
            ClientSecrets = new ClientSecrets()
            {
                ClientId = googleDriveApiSettings.ClientId,
                ClientSecret = googleDriveApiSettings.ClientSecret
            },
            Scopes = new [] { DriveService.Scope.Drive },
            DataStore = new FileDataStore(googleDriveApiSettings.ApplicationName)
        });

        var credential = new UserCredential(apiCodeFlow, googleDriveApiSettings.Username, tokenResponse);

        var service = new DriveService(new BaseClientService.Initializer()
        {
            HttpClientInitializer = credential,
            ApplicationName = googleDriveApiSettings.ApplicationName
        });

        return service;
    }
    
    public async Task<(bool, string)> SaveFileAsync(Stream fileStream, string fileName, string fileType,
        CancellationToken cancellationToken)
    {
        var googleService = AuthenticateGoogleDrive();
        
        var fileMetadata = new Google.Apis.Drive.v3.Data.File()
        {
            Name = fileName,
            MimeType = fileType
        };
        
        var uploadSuccess = false;

        var uploadRequest = googleService.Files.Create(fileMetadata, fileStream, fileType);
        uploadRequest.Fields = "id";

        uploadRequest.ProgressChanged += (progress) =>
        {
            switch (progress.Status)
            {
                case UploadStatus.Uploading:
                {
                    Logger.Info("Uploading File with Name: {FileName} and Id: {FileId} | Progress: {BytesUploaded} bytes", 
                        fileName, uploadRequest.ResponseBody.Id, progress.BytesSent);
                    break;
                }
                case UploadStatus.Completed:
                {
                    Logger.Info("File with Name: {FileName} and Id: {FileId} uploaded successfully", 
                        fileName, uploadRequest.ResponseBody.Id);
                    uploadSuccess = true;
                    break;
                }
                case UploadStatus.Failed:
                {
                    Logger.Info("Upload failed for file with Name: {FileName} and Id: {FileId} | Error message: {ErrorMessage}",
                        fileName, uploadRequest.ResponseBody.Id, progress.Exception.Message);
                    break;
                }
            }
        };

        Logger.Info("Uploading file to Google Drive started. Name: {FileName}", fileName);
        await uploadRequest.UploadAsync(cancellationToken).ConfigureAwait(false);

        return (uploadSuccess, uploadRequest.ResponseBody.Id);
    }

    public async Task<bool> GetFileAsync(string fileId, string savePath, 
        CancellationToken cancellationToken)
    {
        var googleService = AuthenticateGoogleDrive();

        var downloadRequest = googleService.Files.Get(fileId);
        downloadRequest.MediaDownloader.ChunkSize = 256 * 1024;
        
        var downloadSuccess = false;

        downloadRequest.MediaDownloader.ProgressChanged += (progress) =>
        {
            switch (progress.Status)
            {
                case DownloadStatus.Downloading:
                {
                    Logger.Info("Downloading progress: {BytesDownloaded} bytes", progress.BytesDownloaded);
                    break;
                }
                case DownloadStatus.Completed:
                {
                    Logger.Info("File with Id: {FileId} downloaded successfully", fileId);
                    downloadSuccess = true;
                    break;
                }
                case DownloadStatus.Failed:
                {
                    Logger.Info("Download failed for file with Id: {FileId} | Error message: {ErrorMessage}", fileId,
                        progress.Exception.Message);
                    downloadSuccess = false;
                    break;
                }
            }
        };

        //await using var fileStream = new FileStream(savePath, FileMode.Create, FileAccess.Write);
        await using var memoryStream = new MemoryStream();
        await downloadRequest.DownloadAsync(memoryStream, cancellationToken);
        
        //TODO: Modify file saving according to future requirements

        return downloadSuccess;
    }

}