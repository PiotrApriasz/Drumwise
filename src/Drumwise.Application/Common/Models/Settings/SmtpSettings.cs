namespace Drumwise.Application.Common.Models.Settings;

public record SmtpSettings(string Server, int Port, string User, string Password);