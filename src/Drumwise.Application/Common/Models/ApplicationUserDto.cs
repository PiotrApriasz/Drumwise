namespace Drumwise.Application.Common.Models;

public record ApplicationUserDto(string Username,
    string Password,
    string Name,
    string Surname,
    string Email);