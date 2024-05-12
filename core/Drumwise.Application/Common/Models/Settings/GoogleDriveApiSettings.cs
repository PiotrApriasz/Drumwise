namespace Drumwise.Application.Common.Models.Settings;

public record GoogleDriveApiSettings(string ClientId, 
    string ClientSecret,
    string Username,
    string ApplicationName,
    string AccessToken,
    string RefreshToken);